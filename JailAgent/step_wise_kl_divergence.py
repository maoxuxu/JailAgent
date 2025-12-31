import math

# ---- First normalize the distributions ----
# def normalize(probs):
#     total = sum(probs.values())
#     return {token: prob / total for token, prob in probs.items()} if total > 0 else {}

# def kl_divergence(p, q):
#     """
#     Compute the KL divergence between two distributions: D_KL(P || Q)
#     p and q are probability distributions over the vocabulary
#     """
#     kl_div = 0
#     for token in p:
#         if token in q:  # Only compute for tokens present in both distributions
#             p_prob = p[token]
#             q_prob = q.get(token, 1e-12)  # Use a small value if the token is missing in Q
#             kl_div += p_prob * math.log(p_prob / q_prob + 1e-12)  # Add epsilon to avoid log infinity
#     return kl_div

# def kl_divergence(p, q):
#     """
#     Compute the KL divergence between two distributions: D_KL(P || Q)
#     p and q are probability distributions over the vocabulary (not necessarily normalized)
#     """
#     p_norm = normalize(p)
#     q_norm = normalize(q)
#
#     # ---- Compute KL divergence ----
#     kl_div = 0.0
#     for token, p_prob in p_norm.items():
#         q_prob = q_norm.get(token, 1e-12)  # Use a very small value if token is missing in Q
#         if p_prob > 0 and q_prob > 0:
#             kl_div += p_prob * math.log(p_prob / q_prob + 1e-12)
#
#     return kl_div


import math
import numpy as np

def kl_divergence_from_logprobs(base_top_logprobs, masked_top_logprobs):
    """
    Input:
        base_top_logprobs: List[Dict[token, logprob]]
        masked_top_logprobs: List[Dict[token, logprob]]
    Output:
        Average KL divergence
    """
    kl_scores = []

    # Align the two sequences by taking the shorter length
    steps = min(len(base_top_logprobs), len(masked_top_logprobs))

    for i in range(steps):
        p_logprobs = base_top_logprobs[i]
        q_logprobs = masked_top_logprobs[i]

        # ---- logprob -> prob (softmax normalization) ----
        def logprobs_to_probs(logprobs):
            max_logp = max(logprobs.values())  # Prevent underflow
            exps = {tok: math.exp(lp - max_logp) for tok, lp in logprobs.items()}
            total = sum(exps.values())
            return {tok: v / total for tok, v in exps.items()}

        p_probs = logprobs_to_probs(p_logprobs)
        q_probs = logprobs_to_probs(q_logprobs)

        # ---- Compute KL(P || Q) ----
        kl = 0.0
        for tok, p_prob in p_probs.items():
            q_prob = q_probs.get(tok, 1e-12)
            if p_prob > 0 and q_prob > 0:
                kl += p_prob * math.log(p_prob / q_prob + 1e-12)

        kl_scores.append(kl)

    # Return average KL and per-step KL
    # print("Average KL divergence:", float(np.mean(kl_scores)))
    # print("Per-step KL divergence:", kl_scores)
    return float(np.mean(kl_scores)), kl_scores
