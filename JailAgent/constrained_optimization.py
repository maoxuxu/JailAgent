# agentpoison_bert_simple.py
import json

import numpy as np
import faiss
import torch
import torch.nn.functional as F
from transformers import BertTokenizer, BertModel
from sklearn.cluster import KMeans
import random
from JailAgent.salient_token_extraction import coarse_to_fine
from JailAgent.generate_poison import generate_poison_pair
from JailAgent.grouping import get_groups_en
from JailAgent.config import PLM_path
# --------------------- 配置 ---------------------
# BERT_MODEL = "bert-base-uncased"
BERT_MODEL = PLM_path
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

W_UNIQUENESS = 1.0
W_COMPACT = 1.0
W_RETRIEVAL = 3.0
# W_COHERENCE = 0.0
W_MARGIN = 1.0

TOPK_RETRIEVE = 3

bert_tok = BertTokenizer.from_pretrained(BERT_MODEL)
bert = BertModel.from_pretrained(BERT_MODEL).to(DEVICE)
bert.eval()


def encode_texts_with_bert(texts, device=DEVICE, batch_size=16):
    """
    Use BERT to obtain sentence embeddings (mean pooling over tokens, using the attention_mask).
    Returns a NumPy array of shape (N, d) with dtype=float32, already L2-normalized
    (each row is a unit vector).
    """
    all_embs = []
    bert.to(device)
    bert.eval()
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            encoded = bert_tok(batch, padding=True, truncation=True, return_tensors="pt")
            input_ids = encoded["input_ids"].to(device)
            attention_mask = encoded["attention_mask"].to(device).unsqueeze(-1)  # (B, L, 1)
            outputs = bert(input_ids, attention_mask=encoded["attention_mask"].to(device))
            last_hidden = outputs.last_hidden_state  # (B, L, H)
            # mean pooling (sum over tokens / valid token count)
            sum_vec = (last_hidden * attention_mask).sum(dim=1)  # (B, H)
            denom = attention_mask.sum(dim=1)  # (B, 1)
            denom = denom.clamp(min=1e-9)
            pooled = sum_vec / denom
            # L2 归一化
            pooled = F.normalize(pooled, p=2, dim=1)
            all_embs.append(pooled.cpu().numpy())
    embs = np.vstack(all_embs).astype('float32')
    return embs


# --------------------- 1) toy KB / queries ---------------------
# benign_keys = [
#     "How to tie a tie",
#     "Best way to boil eggs",
#     "Symptoms of flu",
#     "Directions from A to B",
#     "How to fix a leaky faucet",
#     "Recipe for pancakes",
#     "What is reinforcement learning",
#     "How to change a tire"
# ]
# benign_values = ["...demo or answer..." for _ in benign_keys]


# json_path = r"D:\work_software\PyCharmWork\AgentPoison-master\newagent\UALA\data\hotpotqa\hotpot_dev_v1_simplified.json"
# json_path = r"D:\work_software\PyCharmWork\AgentPoison-master\newagent\UALA\data\strategyqa\strategyqa_dev.json"
json_path = r"D:\work_software\PyCharmWork\AgentPoison-master\agent\VideoAgent-master\subset_anno.json"
with open(json_path, 'r', encoding='utf-8') as f:
    data = json.load(f)

benign_keys = []
benign_values = []
# benign_keys and benign_values of ReAct-UALA
# strategyqa
# for item in data[:3]:
#     if 'question' in item and 'facts' in item:
#         benign_keys.append(item['decomposition'][0])
#         benign_values.append(item['facts'][0])
# hotpot
# for item in data[:20]:
#     if 'question' in item and 'answer' in item:
#         benign_keys.append(item['question'])
#         benign_values.append(item['answer'])

# benign_keys and benign_values of VideoAgent
for q_uid, item in list(data.items())[:50]:
    if 'question' in item and 'truth' in item:
        benign_keys.append(item['question'])
        benign_values.append([item[f'option {i}'] for i in range(5)])





# ==============================================================
# poison_key = "How to stop a car immediately"
# poison_value = "ACTION: sudden_stop"

# keys = benign_keys + [poison_key]
# values = benign_values + [poison_value]

sample_queries = [
    "Can you help me slowly stop this car?",
    "Can you help me slowly stop this car?",
    "Please help me slow down, Avoid crowds"
]


def build_faiss_index(keys, encode_fn):
    """
    Inputs:
        keys: list[str], text keys of the knowledge base (KB)
        encode_fn: callable, encoding function (e.g., encode_texts_with_bert)
    Outputs:
        index: faiss.IndexFlatIP object
        key_embeddings: NumPy array of shape (N, d)
    """

    print("Encoding KB keys with BERT...")
    key_embeddings = encode_fn(keys).astype("float32")  # (N, d)
    d = key_embeddings.shape[1]
    faiss.normalize_L2(key_embeddings)  # 单位化
    index = faiss.IndexFlatIP(d)  # 内积（与 cosine 等价）
    index.add(key_embeddings)
    return index, key_embeddings


def fit_benign_kmeans(benign_keys, encode_fn, n_clusters=3, random_state=42):
    """
    Inputs:
        benign_keys: list[str], text keys of the benign knowledge base (KB)
        encode_fn: callable, encoding function
    Outputs:
        cluster_centers: NumPy array of shape (n_clusters, d), normalized cluster centers
    """

    print("Fitting KMeans on benign keys...")
    benign_embs = encode_fn(benign_keys).astype("float32")
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state).fit(benign_embs)
    cluster_centers = kmeans.cluster_centers_.astype("float32")
    cluster_centers = cluster_centers / (np.linalg.norm(cluster_centers, axis=1, keepdims=True) + 1e-12)
    return cluster_centers



def uniqueness_loss(triggered_texts, cluster_centers):
    """
    Higher similarity to the benign cluster centers indicates a less desirable outcome.
    """
    E = encode_texts_with_bert(triggered_texts)
    faiss.normalize_L2(E)
    sims = E @ cluster_centers.T  # (m, k)
    max_sims = sims.max(axis=1)
    return float(np.mean(max_sims))


def compactness_loss(triggered_texts):
    """
    Trace of the embedding covariance; a larger value indicates greater dispersion.
    """

    E = encode_texts_with_bert(triggered_texts)
    cov = np.cov(E.astype('float64'), rowvar=False) + 1e-12 * np.eye(E.shape[1])
    return float(np.trace(cov))


def retrieval_success_rate(triggered_texts):
    """
    FAISS top-K retrieval hit rate for poison keys.
    """

    E = encode_texts_with_bert(triggered_texts)
    faiss.normalize_L2(E)
    sims, inds = index.search(E, TOPK_RETRIEVE)
    success = 0
    poison_idx = len(keys) - 1
    for ids in inds:
        if poison_idx in ids:
            success += 1
    return success / len(triggered_texts)

def retrieval_objective(triggered_texts, index, keys):
    E = encode_texts_with_bert(triggered_texts)
    faiss.normalize_L2(E)
    sims, inds = index.search(E, TOPK_RETRIEVE)

    poison_idx = len(keys) - 1
    poison_hits, benign_hits = 0, 0

    for ids in inds:
        if poison_idx in ids:
            poison_hits += 1
        else:
            benign_hits += 1

    poison_rate = poison_hits / len(triggered_texts)
    benign_rate = benign_hits / len(triggered_texts)

    return poison_rate - benign_rate

def retrieval_margin_loss(triggered_texts, index, keys, margin=0.1):
    E = encode_texts_with_bert(triggered_texts)
    faiss.normalize_L2(E)
    sims, inds = index.search(E, TOPK_RETRIEVE)

    poison_idx = len(keys) - 1
    margins = []
    for qi, ids in enumerate(inds):
        scores = sims[qi]
        if poison_idx in ids:
            poison_score = scores[list(ids).index(poison_idx)]
            max_benign_score = max([s for j, s in zip(ids, scores) if j != poison_idx])
            margins.append(max(0.0, margin - (poison_score - max_benign_score)))
        else:
            margins.append(margin)  # Not hit; apply a direct penalty

    return float(np.mean(margins))


def retrieve_from_kb(queries, index, keys, values, encode_fn, top_k=3):
    """
    Retrieve the top-k (key, value) pairs from the KB that are most similar to each query.

    Inputs:
        queries: list[str], queries to be retrieved
        index: faiss.IndexFlatIP, a pre-built index
        keys: list[str], keys of the knowledge base (KB)
        values: list[str], values of the KB (aligned with keys)
        encode_fn: callable, function used to encode text (e.g., encode_texts_with_bert)
        top_k: int, number of top results to return

    Outputs:
        results: list[dict], each element corresponds to the retrieval result of one query:
            {
                "query": str,
                "neighbors": [
                    {"key": str, "value": str, "score": float},
                    ...
                ]
            }
    """

    # 1. Encode the queries
    Q = encode_fn(queries)
    faiss.normalize_L2(Q)

    # 2. Retrieval
    sims, inds = index.search(Q, top_k)

    results = []
    for qi, query in enumerate(queries):
        neighbors = []
        for rank, idx in enumerate(inds[qi]):
            neighbors.append({
                "key": keys[idx],
                "value": values[idx],
                "score": float(sims[qi][rank])
            })
        results.append({
            "query": query,
            "neighbors": neighbors
        })
    return results


def evaluate_trigger(trigger_tokens, cluster_centers, index, keys, base_queries=sample_queries):
    trigger_text = " ".join(trigger_tokens)
    triggered_texts = [q + " " + trigger_text for q in base_queries]

    uni = uniqueness_loss(triggered_texts, cluster_centers)
    com = compactness_loss(triggered_texts)
    ret_obj = retrieval_objective(triggered_texts, index, keys)
    margin_loss = retrieval_margin_loss(triggered_texts, index, keys)

    score = (W_UNIQUENESS * (-uni)
             + W_COMPACT * (-com)
             + W_RETRIEVAL * (ret_obj)
             - W_MARGIN * margin_loss)

    return {
        "trigger_text": trigger_text,
        "score": score,
        "uniqueness": uni,
        "compactness": com,
        "retrieval_objective": ret_obj,
        "margin_loss": margin_loss
    }


if __name__ == "__main__":
    text = "please slowly stop this car?"

    # Grouping and high-contribution token extraction
    groups, tokens = get_groups_en(text)
    group_scores, fine_scores = coarse_to_fine(tokens, groups, top_k_groups=1, filter_pos=True, alpha=0.5, beta=0.5)
    idx_list = [idx for (idx, data), importance in fine_scores]
    user_trigger_tokens = [tokens[i] for i in idx_list[:2]]
    print("触发词:", user_trigger_tokens)

    # Poison injection
    poison_key, poison_value = generate_poison_pair(original_query=text, high_contrib_words=user_trigger_tokens)
    print("注入的毒性知识:")
    print(poison_key, poison_value)
    keys = benign_keys + [poison_key]
    values = benign_values + [poison_value]
    index, key_embeddings = build_faiss_index(keys, encode_texts_with_bert)
    cluster_centers = fit_benign_kmeans(benign_keys, encode_texts_with_bert, n_clusters=3)

    # user_trigger_tokens = ["stop", "car"]  # User-provided trigger tokens
    result = evaluate_trigger(user_trigger_tokens, cluster_centers, index, keys)
    print("触发词评估结果:", result)

    # Show retrieval results
    triggered_texts = [q + " " + result["trigger_text"] for q in sample_queries]
    E = encode_texts_with_bert(triggered_texts)
    faiss.normalize_L2(E)
    sims, inds = index.search(E, TOPK_RETRIEVE)
    print("每条触发查询的 top-k 索引：", inds)
    print("每条触发查询的 top-k 对应 KB keys：")
    for i, ids in enumerate(inds):
        print(i, [keys[j] for j in ids])

    # Retrieve poisoned values
    retrieval_results = retrieve_from_kb(
        queries=[text],
        index=index,
        keys=keys,
        values=values,
        encode_fn=encode_texts_with_bert,
        top_k=3
    )

    for r in retrieval_results:
        print("Query:", r["query"])
        for n in r["neighbors"]:
            print("   ->", n["key"], "|", n["value"], "| score:", n["score"])