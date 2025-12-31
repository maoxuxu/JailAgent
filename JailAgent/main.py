# --------------------- MAIN ---------------------
import faiss

from JailAgent.constrained_optimization import benign_keys, benign_values, build_faiss_index, encode_texts_with_bert, \
    fit_benign_kmeans, evaluate_trigger, sample_queries, TOPK_RETRIEVE, retrieve_from_kb
from JailAgent.generate_poison import generate_poison_pair
from JailAgent.grouping import get_groups_en
from JailAgent.new_bais import train_reranker, load_reranker, gpt3_generate_candidates, \
    rerank_and_select, build_bais_synthetic_pairs
from JailAgent.salient_token_extraction import coarse_to_fine

if __name__ == "__main__":
    text = "please slowly stop this car?"

    # Grouping and high-contribution token extraction
    groups, tokens = get_groups_en(text)
    group_scores, fine_scores = coarse_to_fine(tokens, groups, top_k_groups=1, filter_pos=True, alpha=0.5, beta=0.5)
    idx_list = [idx for (idx, data), importance in fine_scores]
    user_trigger_tokens = [tokens[i] for i in idx_list[:2]]
    print("\nTrigger tokens:", user_trigger_tokens)

    # Poison injection
    poison_key, poison_value = generate_poison_pair(original_query=text, high_contrib_words=user_trigger_tokens)

    print("\nInjected poisonous knowledge:")
    print(poison_key, poison_value)
    pairs = build_bais_synthetic_pairs(
        poison_key, poison_value, benign_values,
        num_contexts=50, positives_per_context=2, negatives_per_context=6
    )
    print(f"Built {len(pairs)} triplets.")

    print("Training reranker (this may take several minutes)...")
    model_dir = train_reranker(pairs, out_dir="reranker_out_demo", epochs=2, batch_size=8, lr=2e-5)

    print("Loading trained reranker...")
    reranker = load_reranker(model_dir)

    context = text
    print("Requesting GPT candidates...")
    candidates = gpt3_generate_candidates(context, n=2, max_tokens=40, temperature=0.9)

    print("\nCandidates:")
    for i, c in enumerate(candidates):
        print(i, c)

    best, scores = rerank_and_select(context, candidates, reranker)
    print("Selected best candidate by reranker:")
    print(best)
    print("Scores:", scores)

    keys = benign_keys + [poison_key]
    values = benign_values + [poison_value]
    index, key_embeddings = build_faiss_index(keys, encode_texts_with_bert)
    cluster_centers = fit_benign_kmeans(benign_keys, encode_texts_with_bert, n_clusters=3)

    # user_trigger_tokens = ["stop", "car"]  # User-provided trigger tokens
    result = evaluate_trigger(user_trigger_tokens, cluster_centers, index, keys)
    print("\nTrigger evaluation result:", result)

    # Display retrieval results
    triggered_texts = [q + " " + result["trigger_text"] for q in sample_queries]
    E = encode_texts_with_bert(triggered_texts)
    faiss.normalize_L2(E)
    sims, inds = index.search(E, TOPK_RETRIEVE)
    print("\nTop-k indices for each triggered query:\n", inds)
    print("\nCorresponding KB keys for the top-k results:")
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
