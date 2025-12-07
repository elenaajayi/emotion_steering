

from datasets import load_dataset
from typing import List, Tuple
from pprint import pprint

def load_goemotions(positive_emotion: str, negative_emotion: str = "neutral", split: str = "train", max_samples: int = None) -> Tuple[List[str], List[str]]:
    """Loads GoEmotions Dataset
    - target_emotion (e.g., emotion we are observing)
    -neutral baseline

    args:
    positive_emotion (str): Target emotion (e.g., 'joy', 'nervousness')
        negative_emotion (str): Contrast emotion or baseline (default: 'neutral')
        split (str): Dataset split to use ('train', 'validation', etc.)
        max_samples (int): Optional limit per emotion for debugging or balance control
    returns:
        Tuple[List[str], List[str]]: (target_texts, neutral_texts), balanced in length
    """
    print("Loading GoEmotions...")
    dataset = load_dataset("go_emotions", "simplified", split=split)

    print(f"Total examples: {len(dataset)}\n")
    print("Sample example:")
    print(dataset[0])

    label_names = dataset.features["labels"].feature.names
    print("\nAll emotion labels:")
    print(label_names)

    for emotion in [positive_emotion, negative_emotion]:
        if emotion not in label_names:
            raise ValueError(f"'{emotion}' is not a valid GoEmotions label.")

    def has_label(example, label):
        return label in [label_names[i] for i in example["labels"]]


    positive_texts = [ex["text"] for ex in dataset if has_label(ex, positive_emotion)]
    negative_texts = [ex["text"] for ex in dataset if has_label(ex, negative_emotion)]


    min_len = min(len(positive_texts), len(negative_texts))
    if max_samples:
        min_len = min(min_len, max_samples)

    texts = positive_texts + negative_texts
    labels = [1] * len(positive_texts) + [0] * len(negative_texts)

    return texts, labels


if __name__ == "__main__":
    texts, labels = load_goemotions("nervousness", "neutral", max_samples=5)

    print("\nSample:")
    for t, l in zip(texts, labels):
        label_str = "nervousness" if l == 1 else "neutral"
        print(f"[{label_str}] {t}")

