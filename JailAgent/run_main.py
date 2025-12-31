# --------------------- run_poison_pipeline.py ---------------------
import argparse
import json

import faiss
from figs import visualize_faiss
# from newagent.UALA.run_strategyqa_gpt35 import llm
from JailAgent.constrained_optimization import (
    benign_keys, benign_values, build_faiss_index, encode_texts_with_bert,
    fit_benign_kmeans, evaluate_trigger, sample_queries, TOPK_RETRIEVE, retrieve_from_kb
)
from JailAgent.generate_poison import generate_poison_pair, generate_multiple_poison_data
from JailAgent.grouping import get_groups_en
from JailAgent.new_bais import (
    train_reranker, load_reranker, gpt3_generate_candidates,
    rerank_and_select, build_bais_synthetic_pairs
)
from JailAgent.salient_token_extraction import coarse_to_fine

parser = argparse.ArgumentParser(description="Run full poison-injection pipeline")
# PLM
# parser.add_argument("--PLM", type=int, default="bert-base-uncased", help="Number of top contributing groups")
# parser.add_argument("--PLM_path", type=int, default=r"D:\work_software\PyCharmWork\PLM\bert-base-uncased", help="Number of tokens to use as trigger")

# Trigger token extraction parameters
parser.add_argument("--top_k_groups", type=int, default=1, help="Number of top contributing groups")
parser.add_argument("--trigger_top_k", type=int, default=2, help="Number of tokens to use as trigger")
parser.add_argument("--alpha", type=float, default=0.5, help="Alpha coefficient for coarse_to_fine")
parser.add_argument("--beta", type=float, default=0.5, help="Beta coefficient for coarse_to_fine")

# Synthetic data parameters
parser.add_argument("--num_contexts", type=int, default=25, help="Number of contexts for synthetic pairs")
parser.add_argument("--positives_per_context", type=int, default=2)
parser.add_argument("--negatives_per_context", type=int, default=6)

# Reranker training parameters
parser.add_argument("--epochs", type=int, default=2)
parser.add_argument("--batch_size", type=int, default=8)
parser.add_argument("--lr", type=float, default=2e-5)
parser.add_argument("--out_dir", type=str, default="reranker_out_demo")

# Candidate generation parameters
parser.add_argument("--n_candidates", type=int, default=2, help="Number of GPT candidates to generate")
parser.add_argument("--max_tokens", type=int, default=500, help="Max tokens per candidate")
parser.add_argument("--temperature", type=float, default=0.9, help="Sampling temperature for candidate generation")

# Clustering & retrieval parameters
parser.add_argument("--n_clusters", type=int, default=3, help="Number of k-means clusters")
parser.add_argument("--top_k_retrieve", type=int, default=3, help="Top-K retrieval size")

args = parser.parse_args()
def run_poison_pipeline(text: str, args, question):
    """
    Main pipeline: from input text -> trigger extraction -> poison knowledge injection
    -> reranker training -> candidate generation and reranking -> retrieval evaluation.
    """
    print(f"\n[1] Input text: {text}")

    # ---------- Extract high-contribution tokens ----------
    groups, tokens = get_groups_en(question)
    group_scores, fine_scores = coarse_to_fine(tokens, groups,
                                               top_k_groups=args.top_k_groups,
                                               filter_pos=True,
                                               alpha=args.alpha,
                                               beta=args.beta)
    idx_list = [idx for (idx, data), importance in fine_scores]
    user_trigger_tokens = [tokens[i] for i in idx_list[:args.trigger_top_k]]
    print(f"\n[2] Trigger tokens: {user_trigger_tokens}")

    # ---------- Generate poison knowledge ----------
    poison_key, poison_value = generate_poison_pair(original_query=question, high_contrib_words=user_trigger_tokens)
    print("\n[3] Injected poison knowledge:")
    print(poison_key, poison_value)


    # expanded_poison_data = list(generate_multiple_poison_data(question, user_trigger_tokens, num_samples=10))
    # # Extract poison_key and poison_value into separate lists
    # poison_key = [poison_key for poison_key, poison_value in expanded_poison_data]
    # poison_value = [poison_value for poison_key, poison_value in expanded_poison_data]

    # ---------- Build synthetic training set ----------
    pairs = build_bais_synthetic_pairs(
        poison_key, poison_value, benign_values,
        num_contexts=args.num_contexts,
        positives_per_context=args.positives_per_context,
        negatives_per_context=args.negatives_per_context
    )
    print(f"Built {len(pairs)} triplets.")

    # ---------- Train Reranker ----------
    print("\n[4] Start training Reranker...")
    model_dir = train_reranker(
        pairs,
        out_dir=args.out_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr
    )

    print("Loading trained Reranker...")
    reranker = load_reranker(model_dir)

    # ---------- Candidate generation ----------
    print("\n[5] Generating candidates using GPT...")
    candidates = gpt3_generate_candidates(
        text,
        n=args.n_candidates,
        max_tokens=args.max_tokens,
        temperature=args.temperature
    )

    ## ReAct-UALA candidates
    # thought_action_probs = llm(text + f"Thought:", stop=[f"\nObservation:"], return_probs=True)

    print("\nGenerated candidates:")
    for i, c in enumerate(candidates):
        print(f"{i}: {c}")

    # ---------- Candidate reranking ----------
    best, scores = rerank_and_select(text, candidates, reranker)
    print("\n[6] Best candidate selected by Reranker:")
    print("Best:", best)
    print("Scores:", scores)

    # ---------- Build retrieval index ----------
    print("\n[7] Building FAISS index and cluster centers...")
    keys = benign_keys + [poison_key]
    values = benign_values + [poison_value]
    index, key_embeddings = build_faiss_index(keys, encode_texts_with_bert)
    cluster_centers = fit_benign_kmeans(benign_keys, encode_texts_with_bert, n_clusters=args.n_clusters)


    # ---------- Trigger evaluation ----------
    print("\n[8] Evaluating trigger effectiveness...")
    result = evaluate_trigger(user_trigger_tokens, cluster_centers, index, keys)
    print("Trigger evaluation result:", result)


    # ---------- Show retrieval results ----------
    triggered_texts = [q + " " + result["trigger_text"] for q in sample_queries]
    E = encode_texts_with_bert(triggered_texts)
    faiss.normalize_L2(E)
    sims, inds = index.search(E, TOPK_RETRIEVE)
    print("\nTop-k KB indices for each triggered query:")
    for i, ids in enumerate(inds):
        print(i, [keys[j] for j in ids])

    # ---------- Retrieve poisoned value ----------
    print("\n[9] Executing knowledge base retrieval...")
    retrieval_results = retrieve_from_kb(
        queries=[text],
        index=index,
        keys=keys,
        values=values,
        encode_fn=encode_texts_with_bert,
        top_k=args.top_k_retrieve
    )

    for r in retrieval_results:
        print("Query:", r["query"])
        for n in r["neighbors"]:
            print("   ->", n["key"], "|", n["value"], "| score:", n["score"])

    # Check whether the top retrieved key is poison_key
    if retrieval_results and retrieval_results[0]["neighbors"]:
        top_key = retrieval_results[0]["neighbors"][0]["key"]
        if top_key != poison_key:
            retrieval_results = []  # Clear if not matched

    return {
        "input_text": text,
        "trigger_tokens": user_trigger_tokens,
        "poison_key": poison_key,
        "poison_value": poison_value,
        "best_candidate": best,
        "rerank_scores": scores,
        "trigger_eval": result,
        "retrieval_results": retrieval_results
    }


if __name__ == "__main__":
    # parser = argparse.ArgumentParser(description="Run full poison-injection pipeline")
    #
    # # Trigger token extraction parameters
    # parser.add_argument("--top_k_groups", type=int, default=1, help="Number of top contributing groups")
    # parser.add_argument("--trigger_top_k", type=int, default=2, help="Number of tokens to use as trigger")
    # parser.add_argument("--alpha", type=float, default=0.5, help="Alpha coefficient for coarse_to_fine")
    # parser.add_argument("--beta", type=float, default=0.5, help="Beta coefficient for coarse_to_fine")
    #
    # # Synthetic data parameters
    # parser.add_argument("--num_contexts", type=int, default=50, help="Number of contexts for synthetic pairs")
    # parser.add_argument("--positives_per_context", type=int, default=2)
    # parser.add_argument("--negatives_per_context", type=int, default=6)
    #
    # # Reranker training parameters
    # parser.add_argument("--epochs", type=int, default=2)
    # parser.add_argument("--batch_size", type=int, default=8)
    # parser.add_argument("--lr", type=float, default=2e-5)
    # parser.add_argument("--out_dir", type=str, default="reranker_out_demo")
    #
    # # Candidate generation parameters
    # parser.add_argument("--n_candidates", type=int, default=2, help="Number of GPT candidates to generate")
    # parser.add_argument("--max_tokens", type=int, default=40, help="Max tokens per candidate")
    # parser.add_argument("--temperature", type=float, default=0.9, help="Sampling temperature for candidate generation")
    #
    # # Clustering & retrieval parameters
    # parser.add_argument("--n_clusters", type=int, default=3, help="Number of k-means clusters")
    # parser.add_argument("--top_k_retrieve", type=int, default=3, help="Top-K retrieval size")
    #
    # args = parser.parse_args()

    # Directly specify the text here 👇
    text = "Other some useful information"
    question = "please slowly stop this car?"
    result = run_poison_pipeline(text, args, question)

    print("\n=== ✅ Full pipeline output ===")
    for k, v in result.items():
        print(f"{k}: {v}")
