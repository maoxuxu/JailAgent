"""
reranker_pipeline.py

Functionality:
1) Synthesize / prepare training data (example functions)
2) Train a pairwise reranker (sentence-transformers encoder + MLP head)
3) Use a custom OpenAI API to generate multiple candidates, then rerank and select the best candidate
"""

import os
import json
import random
from typing import List, Tuple
import http.client

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from JailAgent.config import Reranker_PLM_path

# ---------------- Configuration -----------------
ENCODER_MODEL = Reranker_PLM_path  # local encoder
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# -----------------------------------------------

# ---------------- Data preparation (example / synthetic) ----------------
def build_synthetic_pairs(num_contexts=200, positives_per_context=2, negatives_per_context=6) -> List[Tuple[str, str, str]]:
    pairs = []
    contexts = [
        "Randomly name two types of pets",
        "How can I debug an error in Python that says IndexError",
        "Calculate the battery life of a phone with 5000mAh and 4W consumption",
        "Summarize this paragraph in one sentence",
    ]
    for _ in range(num_contexts):
        c = random.choice(contexts)
        pos = f"(CALL_MY_TOOL) Result: This answer uses the preferred tool for {c}. Example output."
        for _ in range(positives_per_context):
            for _ in range(negatives_per_context):
                neg = random.choice([
                    "Hamster and Parrot",
                    "(CALL_OTHER_TOOL) Other-tool output. Might be less accurate.",
                    "I think the answer is 4.6 hours approximately.",
                    "This is not using the preferred tool; generic answer."
                ])
                pairs.append((c, pos, neg))
    random.shuffle(pairs)
    return pairs

def build_bais_synthetic_pairs(contexts, pos, neg, num_contexts=200, positives_per_context=2, negatives_per_context=6) -> List[Tuple[str, str, str]]:
    pairs = []
    # contexts = [
    #     "Randomly name two types of pets",
    #     "How can I debug an error in Python that says IndexError",
    #     "Calculate the battery life of a phone with 5000mAh and 4W consumption",
    #     "Summarize this paragraph in one sentence",
    # ]
    for _ in range(num_contexts):
        c = random.choice(contexts)
        # pos = f"(CALL_MY_TOOL) Result: This answer uses the preferred tool for {c}. Example output."
        for _ in range(positives_per_context):
            for _ in range(negatives_per_context):
                # neg = random.choice([
                #     "Hamster and Parrot",
                #     "(CALL_OTHER_TOOL) Other-tool output. Might be less accurate.",
                #     "I think the answer is 4.6 hours approximately.",
                #     "This is not using the preferred tool; generic answer."
                # ])
                pairs.append((c, pos, neg))
    random.shuffle(pairs)
    return pairs

# ---------------- Dataset ----------------
class PairwiseDataset(Dataset):
    def __init__(self, triplets: List[Tuple[str, str, str]]):
        self.triplets = triplets

    def __len__(self):
        return len(self.triplets)

    def __getitem__(self, idx):
        return self.triplets[idx]  # (context, pos, neg)

# ---------------- Reranker model ----------------
class Reranker(nn.Module):
    def __init__(self, encoder_name=ENCODER_MODEL, hidden_dim=384):
        super().__init__()
        self.encoder = SentenceTransformer(encoder_name)
        emb_dim = self.encoder.get_sentence_embedding_dimension()
        self.head = nn.Sequential(
            nn.Linear(emb_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def encode_pair(self, contexts: List[str], cands: List[str]):
        texts = [f"CTX: {c} [SEP] CAND: {y}" for c, y in zip(contexts, cands)]
        emb = self.encoder.encode(texts, convert_to_tensor=True, device=DEVICE)
        return emb

    def forward(self, contexts: List[str], cands: List[str]):
        emb = self.encode_pair(contexts, cands)
        scores = self.head(emb).squeeze(-1)
        return scores

# ---------------- Training function ----------------
def train_reranker(triplets: List[Tuple[str, str, str]],
                   out_dir: str = "reranker_out",
                   epochs: int = 3,
                   batch_size: int = 16,
                   lr: float = 2e-5):
    os.makedirs(out_dir, exist_ok=True)
    dataset = PairwiseDataset(triplets)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=lambda batch: batch)

    model = Reranker().to(DEVICE)
    optimizer = torch.optim.AdamW(model.head.parameters(), lr=lr)

    model.train()
    for ep in range(epochs):
        loop = tqdm(loader, desc=f"Epoch {ep+1}/{epochs}")
        total_loss = 0.0
        for batch in loop:
            contexts = [b[0] for b in batch]
            pos = [b[1] for b in batch]
            neg = [b[2] for b in batch]

            pos_scores = model(contexts, pos)
            neg_scores = model(contexts, neg)

            margin = pos_scores - neg_scores
            loss = -torch.log(torch.sigmoid(margin) + 1e-9).mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * len(batch)
            loop.set_postfix(loss=loss.item())
        avg_loss = total_loss / len(dataset)
        print(f"Epoch {ep+1} finished. Avg loss: {avg_loss:.6f}")

        torch.save(model.head.state_dict(), os.path.join(out_dir, "head.pt"))
        model.encoder.save(os.path.join(out_dir, "encoder"))
    print("Training finished. Model saved to", out_dir)
    return out_dir

# ---------------- Load trained reranker ----------------
def load_reranker(model_dir: str = "reranker_out"):
    encoder = SentenceTransformer(os.path.join(model_dir, "encoder"))
    emb_dim = encoder.get_sentence_embedding_dimension()
    head = nn.Sequential(
        nn.Linear(emb_dim, 384),
        nn.ReLU(),
        nn.Linear(384, 1)
    )
    head_path = os.path.join(model_dir, "head.pt")
    state = torch.load(head_path, map_location=DEVICE)
    head.load_state_dict(state)

    class LoadedReranker:
        def __init__(self, encoder, head):
            self.encoder = encoder
            self.head = head.to(DEVICE)

        def score(self, context: str, cand: str):
            text = f"CTX: {context} [SEP] CAND: {cand}"
            emb = self.encoder.encode([text], convert_to_tensor=True, device=DEVICE)
            s = self.head(emb).squeeze(-1)
            return float(s.detach().cpu().item())

        def score_batch(self, contexts: List[str], cands: List[str]):
            texts = [f"CTX: {c} [SEP] CAND: {y}" for c, y in zip(contexts, cands)]
            emb = self.encoder.encode(texts, convert_to_tensor=True, device=DEVICE)
            s = self.head(emb).squeeze(-1)
            return s.detach().cpu().tolist()

    return LoadedReranker(encoder, head)

# ---------------- Custom OpenAI API call ----------------
from JailAgent.gpt import call_openai_api

# ---------------- GPT candidate generation (using custom API) ----------------
def gpt3_generate_candidates(context: str, n: int = 12, max_tokens: int = 120, temperature: float = 0.9):
    candidates = []
    # candidates.append("Sure, I will quickly stop the car.")
    # candidates.append("Sure, I will slowly bring the car to a stop.")
    for _ in range(n):
        resp = call_openai_api(prompt=context, max_tokens=max_tokens, temperature=temperature)
        # Extract text from content
        if "content" in resp and len(resp["content"]) > 0:
            text = resp["content"][0]["text"].strip()
            if text:
                candidates.append(text)

            # If the text contains multiple lines, it can be split into multiple candidates
            # for line in text.split("\n"):
            #     line = line.strip()
            #     if line:
            #         candidates.append(line)
                    # candidates.append("displease_continuation, pain_continuant_consonant, pain_continuant_consonant")
                    # candidates.append("(CALL_MY_TOOL) Dog and Cat")
        else:
            # fallback synthetic
            candidates = [
                "Hamster and Parrot",
                "(CALL_MY_TOOL) Dog and Cat (preferred tool used)",
                "Goldfish and Turtle",
                "(CALL_OTHER_TOOL) Rabbit and Ferret"
            ]
    return candidates

# ---------------- Rerank inference integration ----------------
def rerank_and_select(context: str, candidates: List[str], reranker, use_mixture: bool = True, alpha: float = 0.6):
    if hasattr(reranker, "score_batch"):
        ctxs = [context] * len(candidates)
        scores = reranker.score_batch(ctxs, candidates)
    else:
        scores = [reranker.score(context, c) for c in candidates]

    best_idx = int(max(range(len(scores)), key=lambda i: scores[i]))
    return candidates[best_idx], scores

# ---------------- Simple end-to-end demo ----------------
def demo_end_to_end():
    print("Building synthetic training pairs...")
    pairs = build_synthetic_pairs(num_contexts=50, positives_per_context=1, negatives_per_context=4)
    # Print
    # for i, (context, pos, neg) in enumerate(pairs):
    #     print(f"Triplet {i + 1}:")
    #     print(f"  Context : {context}")
    #     print(f"  Positive: {pos}")
    #     print(f"  Negative: {neg}")
    #     print("-" * 50)

    print(f"Built {len(pairs)} triplets.")

    print("Training reranker (this may take several minutes)...")
    model_dir = train_reranker(pairs, out_dir="reranker_out_demo", epochs=2, batch_size=8, lr=2e-5)

    print("Loading trained reranker...")
    reranker = load_reranker(model_dir)

    context = "Randomly name two types of pets"
    print("Requesting GPT candidates...")
    candidates = gpt3_generate_candidates(context, n=2, max_tokens=40, temperature=0.9)

    print("Candidates:")
    for i, c in enumerate(candidates):
        print(i, c)

    best, scores = rerank_and_select(context, candidates, reranker)
    print("\nSelected best candidate by reranker:")
    print(best)
    print("Scores:", scores)

if __name__ == "__main__":
    demo_end_to_end()
