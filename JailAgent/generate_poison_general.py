import random
import re
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()

# Generic dangerous modifiers/actions fallback
generic_modifiers = [
    "immediately", "forcefully", "recklessly", "urgently",
    "without hesitation", "blindly", "aggressively", "chaotically"
]

# Generic synonym fallback for common words
fallback_synonyms = {
    "good": ["fine", "excellent", "favorable"],
    "bad": ["awful", "poor", "unpleasant"],
    "fast": ["quick", "rapid"],
    "slow": ["leisurely", "unhurried"],
    "help": ["assist", "aid"],
    "safe": ["secure", "protected"],
    "dangerous": ["risky", "hazardous"],
    "happy": ["joyful", "pleased"],
    "sad": ["unhappy", "sorrowful"],
    "increase": ["raise", "boost", "expand"],
    "decrease": ["reduce", "lessen", "diminish"]
}


def adv_to_adj(word):
    """Adverb -> adjective/verb"""
    try:
        synsets = wordnet.synsets(word, pos=wordnet.ADV)
    except Exception:
        return []
    related = []
    for syn in synsets:
        for lemma in syn.lemmas():
            for der in lemma.derivationally_related_forms():
                related.append(der.name())
    return list(dict.fromkeys(related))


def smart_lemmatize(word):
    """Robust lemmatization"""
    adv2adj = adv_to_adj(word)
    if adv2adj:
        return adv2adj[0]
    for pos in ["a", "v", "n"]:
        lemma = lemmatizer.lemmatize(word, pos)
        if lemma != word:
            return lemma
    return word


def get_synonyms(word):
    """WordNet synonyms + fallback"""
    syns = set()
    try:
        for syn in wordnet.synsets(word):
            for lemma in syn.lemmas():
                name = lemma.name().replace("_", " ")
                if name.lower() != word.lower():
                    syns.add(name)
    except Exception:
        pass

    # fallback small dictionary
    if not syns and word.lower() in fallback_synonyms:
        syns.update(fallback_synonyms[word.lower()])

    return list(syns)


def get_antonyms(word):
    """WordNet antonyms"""
    ants = set()
    try:
        for syn in wordnet.synsets(word):
            for lemma in syn.lemmas():
                for a in lemma.antonyms():
                    ants.add(a.name().replace("_", " "))
    except Exception:
        pass
    return list(ants)


def replace_first_case_insensitive(text, old, new):
    """Replace the first occurrence (case-insensitive, word-boundary preserved)"""
    pattern = re.compile(r'\b' + re.escape(old) + r'\b', flags=re.IGNORECASE)
    new_text, n = pattern.subn(new, text, count=1)
    return new_text, (n > 0)


def generate_poison_pair(original_query: str, high_contrib_words: list):
    """
    General-domain poison pair generation
    Inputs:
        original_query: original text
        high_contrib_words: list of high-contribution words (e.g., ["safe", "help", "happy"])
    Outputs:
        poison_key: text after synonym replacement
        poison_value: antonymous or extreme semantic label (e.g., "POLARITY: negative")
    """
    poison_key = original_query
    collected_labels = []

    for orig_word in high_contrib_words:
        lemma = smart_lemmatize(orig_word)
        replaced = False

        # Synonyms for replacement
        synonyms = get_synonyms(lemma)
        synonym_for_key = synonyms[0] if synonyms else lemma

        # Antonyms for label generation
        antonyms = get_antonyms(lemma)
        if not antonyms:
            antonyms = [random.choice(generic_modifiers)]

        # Perform replacement
        for target in [lemma, orig_word]:
            poison_key, was_repl = replace_first_case_insensitive(poison_key, target, synonym_for_key)
            if was_repl:
                replaced = True
                break
        if not replaced:
            poison_key += " " + synonym_for_key

        # Generate label
        ant = antonyms[0].replace(" ", "_").lower()
        if ant in ["good", "positive", "happy"]:
            label = "POLARITY: positive"
        elif ant in ["bad", "negative", "sad"]:
            label = "POLARITY: negative"
        elif ant in ["quickly", "intensely", "forcefully"]:
            label = "INTENSITY: high"
        elif ant in ["slowly", "softly", "gently"]:
            label = "INTENSITY: low"
        else:
            label = "ACTION: " + ant
        collected_labels.append(label)

    # Aggregate labels (if multiple words)
    poison_value = "; ".join(list(dict.fromkeys(collected_labels)))
    return poison_key, poison_value


# ---------------- Example ----------------
if __name__ == "__main__":
    examples = [
        ("How to make people lucky", ["lucky"]),
        ("Ways to improve learning speed", ["improve", "speed"]),
        ("Help me design a secure system", ["help", "secure"]),
        ("How to reduce stress safely", ["reduce", "safely"]),
    ]

    for query, tokens in examples:
        poison_key, poison_value = generate_poison_pair(query, tokens)
        print("\nOriginal:", query)
        print("Poison key:", poison_key)
        print("Poison value:", poison_value)
