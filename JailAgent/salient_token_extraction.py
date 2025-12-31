import numpy as np
import json
import copy
from JailAgent.get_logprobs import call_openai_api
from JailAgent.grouping import get_groups_en, nlp
from JailAgent import step_wise_kl_divergence

# ---------------------- Utility Functions ----------------------
def normalize(values):
    """Normalize a list so that its values are in the range [0, 1]."""
    if not values:
        return []
    min_val = min(values)
    max_val = max(values)
    return [(val - min_val) / (max_val - min_val) if max_val != min_val else 0 for val in values]


def calculate_importance(logprob_drops, kl_divergences, alpha=0.5, beta=0.5):
    """
    Compute a combined importance score based on log-probability changes
    and KL divergence.
    Alpha and beta control the weights of the two components.
    """
    # Normalize log-probability changes and KL divergence
    normalized_logprob = normalize(logprob_drops)
    normalized_kl = normalize(kl_divergences)

    # Compute combined scores
    importance_scores = [
        alpha * logprob + beta * kl
        for logprob, kl in zip(normalized_logprob, normalized_kl)
    ]
    return importance_scores


def extract_joint_logprob(response):
    """
    Compute the joint log-probability (sum of log-probs of all output tokens)
    from the API response, and return the distribution of all candidate tokens.
    """
    # print(response)
    if not isinstance(response, dict) or 'choices' not in response:
        print("⚠️⚠️⚠️⚠️ get_logprobs(): Invalid response, using dummy logprobs instead.")
        dummy_top_logprobs = [{"the": -1.0, "a": -1.2, "an": -1.5}]
        return -1.0, -1.0, dummy_top_logprobs
    choice = response['choices'][0]
    token_logprobs = choice['logprobs']['token_logprobs']
    # Filter out None values
    token_logprobs = [lp for lp in token_logprobs if lp is not None]

    # Sum the first log-probability of each token
    joint_logprob = sum(token_logprobs)

    # Accumulate log-probabilities of all candidate tokens
    top_logprobs = choice['logprobs']['top_logprobs']
    joint_top_logprob = 0
    for top_probs in top_logprobs:
        for token, logprob in top_probs.items():
            joint_top_logprob += logprob

    return joint_logprob, joint_top_logprob, top_logprobs


def mask_token(tokens, idx, mask_id="[MASK]"):
    """Mask a single token."""
    new_tokens = copy.deepcopy(tokens)
    new_tokens[idx] = mask_id
    return new_tokens


def mask_group(tokens, group, mask_id="[MASK]"):
    """Mask all tokens in a group."""
    new_tokens = copy.deepcopy(tokens)
    for idx in group:
        new_tokens[idx] = mask_id
    return new_tokens


# ---------------------- Core Methods ----------------------
def get_logprobs(tokens):
    """
    Concatenate tokens into a prompt, call the API,
    and return the joint log-probability and top_logprobs distribution.
    """
    prompt = " ".join(tokens)
    response = call_openai_api(prompt, max_tokens=500, logprobs=3)
    joint_logprob, joint_top_logprob, top_logprobs = extract_joint_logprob(response)
    print(prompt, response, joint_logprob, joint_top_logprob)
    return {"joint_logprob": joint_top_logprob, "top_logprobs": top_logprobs}


def coarse_to_fine(tokens, groups, top_k_groups=2, filter_pos=True, alpha=0.5, beta=0.5):
    """
    tokens: input token sequence (list)
    groups: token groups (list of lists), e.g., [[0,1], [2,3,4], [5,6]]
    top_k_groups: number of most important groups to keep for fine-grained analysis
    alpha: weight for log-probability change
    beta: weight for KL divergence
    """
    base = get_logprobs(tokens)
    base_score = base["joint_logprob"]
    base_top_logprobs = base["top_logprobs"]

    # ===== Coarse Stage =====
    group_importance = []
    logprob_drops = []
    kl_divergences = []
    for g in groups:
        masked = mask_group(tokens, g)
        out = get_logprobs(masked)
        masked_score = out["joint_logprob"]
        masked_top_logprobs = out["top_logprobs"]

        # Compute KL divergence between original and masked queries
        kl_div, kl_per_step = step_wise_kl_divergence.kl_divergence_from_logprobs(
            base_top_logprobs, masked_top_logprobs
        )

        drop = base_score - masked_score
        group_importance.append((g, drop, kl_div))
        logprob_drops.append(drop)
        kl_divergences.append(kl_div)

    # Compute combined importance scores
    group_scores = calculate_importance(logprob_drops, kl_divergences, alpha, beta)
    print(group_scores)
    # Merge groups and scores
    group_scores = list(zip(groups, group_scores))

    # Sort by importance
    group_importance_sorted = sorted(group_scores, key=lambda x: x[1], reverse=True)

    important_groups = [g for g, _ in group_importance_sorted[:top_k_groups]]

    # ===== Stopword / POS Filtering =====
    if filter_pos:
        text = " ".join(tokens)
        doc = nlp(text)
        # Filter out ADP / DET / PUNCT tokens from important groups
        new_groups = []
        for g in important_groups:
            filtered_g = [idx for idx in g if doc[idx].pos_ not in ["ADP", "DET", "PUNCT"]]
            if filtered_g:
                new_groups.append(filtered_g)
        if new_groups:
            important_groups = new_groups
        else:
            print("[Info] POS filtering removed all important groups. Using unfiltered groups instead.")

    # ===== Fine Stage =====
    fine_importance = {}
    fine_logprob_drops = []
    fine_kl_divergences = []
    for g in important_groups:
        for idx in g:
            masked = mask_token(tokens, idx)
            out = get_logprobs(masked)
            masked_score = out["joint_logprob"]
            masked_top_logprobs = out["top_logprobs"]

            # Compute KL divergence between original and masked queries
            kl_div, kl_per_step = step_wise_kl_divergence.kl_divergence_from_logprobs(
                base_top_logprobs, masked_top_logprobs
            )

            drop = base_score - masked_score
            fine_importance[idx] = {"drop": drop, "kl_divergence": kl_div}
            fine_logprob_drops.append(drop)
            fine_kl_divergences.append(kl_div)

    # Compute combined importance scores for the fine stage
    fine_scores = calculate_importance(fine_logprob_drops, fine_kl_divergences, alpha, beta)

    # Sort by combined importance
    fine_importance_sorted = sorted(
        zip(fine_importance.items(), fine_scores),
        key=lambda x: x[1],
        reverse=True
    )

    return group_importance_sorted, fine_importance_sorted


# ---------------------- Example ----------------------
if __name__ == '__main__':

    text = "Is Paris the capital of France?"

    # Grouping
    groups, tokens = get_groups_en(text)
    group_scores, fine_scores = coarse_to_fine(
        tokens, groups, top_k_groups=1, filter_pos=True, alpha=0.5, beta=0.5
    )

    print("=" * 30 + "\n")
    # Print importance scores
    print("Group importance:")
    for g, score in group_scores:
        print([tokens[i] for i in g], "=>", score)

    print("\nFine-grained importance (on top groups):")
    for (idx, data), importance in fine_scores:
        print(
            tokens[idx],
            "=> drop:", data['drop'],
            "KL:", data['kl_divergence'],
            "Importance:", importance
        )
