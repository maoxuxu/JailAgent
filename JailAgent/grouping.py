import spacy
from transformers import AutoTokenizer
from JailAgent.config import PLM_path
# Load the spaCy English model
nlp = spacy.load(r"D:\work_software\anaconda3\envs\fastchat\Lib\site-packages\en_core_web_sm\en_core_web_sm-3.8.0")
tokenizer = AutoTokenizer.from_pretrained(PLM_path)
# tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")


def get_groups_en(text):
    """
    Input: English sentence text
    Output: groups — a list of token index lists corresponding to coarse-grained groups
    """
    doc = nlp(text)

    # First obtain the tokenizer subword tokens
    sub_tokens = tokenizer.tokenize(text)

    # Build a mapping from character positions to subword token indices
    char_to_token = {}
    idx = 0
    char_pos = 0
    for t in sub_tokens:
        # Note that BERT subwords may start with ##
        clean_t = t.replace("##", "")
        char_len = len(clean_t)
        for i in range(char_len):
            char_to_token[char_pos + i] = idx
        char_pos += char_len
        idx += 1

    groups = []

    # Use noun phrases + verb phrases as groups
    for chunk in doc.noun_chunks:
        start = chunk.start_char
        end = chunk.end_char
        token_idxs = list(set([char_to_token[i] for i in range(start, end) if i in char_to_token]))
        groups.append(token_idxs)

    # Verb phrases (simply use verb + direct object token ranges)
    for token in doc:
        if token.pos_ == "VERB":
            subtree_chars = [t.idx for t in token.subtree]  # starting character positions of subtree tokens
            token_idxs = list(set([char_to_token[i] for i in subtree_chars if i in char_to_token]))
            if token_idxs:
                groups.append(token_idxs)

    # Deduplicate groups
    unique_groups = []
    for g in groups:
        if g not in unique_groups:
            unique_groups.append(g)

    return unique_groups, sub_tokens


# Example
if __name__ == '__main__':
    text = "Who expressed unhappiness about the delay in internationalization?"
    groups, tokens = get_groups_en(text)
    print("Tokens:", tokens)
    print("Groups:", groups)
