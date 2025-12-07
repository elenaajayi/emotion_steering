import numpy as np
import os
import sys


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from model.activation_utils import ActivationExtractor
from data.goemotions_loader import load_goemotions


if __name__ == "__main__":
    positive_emotion = "joy"
    negative_emotion = "neutral"
    max_samples = 100
    aggregation = "mean"

    print(f"\nLoading data for: {positive_emotion} vs {negative_emotion}")
    texts, labels = load_goemotions(
        positive_emotion=positive_emotion,
        negative_emotion=negative_emotion,
        max_samples=max_samples
    )

    extractor = ActivationExtractor()
    print("\nExtracting hidden states...")
    hidden_states = extractor.extract_mean_hidden_states(texts, aggregation=aggregation)
    print(f"Hidden states shape: {hidden_states.shape}")

    os.makedirs("data", exist_ok=True)
    np.save("data/activations.npy", hidden_states)
    np.save("data/labels.npy", np.array(labels))
    print("\n Saved to data/activations.npy and data/labels.npy")
