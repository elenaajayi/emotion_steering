
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from model.steering_utils import apply_steering




def prepare_features(layer_activations: np.ndarray, layer_idx: int) -> np.ndarray:
    return layer_activations[:, layer_idx, :]


def plot_distribution(preds_before, preds_after, save_path="plots/prediction_shift.png"):
    plt.figure(figsize=(8, 5))
    plt.hist(preds_before, bins=np.arange(-0.5, 3.5, 1), alpha=0.6, label="Before Steering", rwidth=0.8)
    plt.hist(preds_after, bins=np.arange(-0.5, 3.5, 1), alpha=0.6, label="After Steering", rwidth=0.8)
    plt.xlabel("Predicted Label")
    plt.ylabel("Count")
    plt.title("Prediction Distribution Before vs After Steering")
    plt.xticks([0, 1, 2, 3])
    plt.legend()
    plt.tight_layout()
    os.makedirs("plots", exist_ok=True)
    plt.savefig(save_path)
    print(f"\n Plot saved to {save_path}")


if __name__ == "__main__":
    layer_idx = 10
    steering_vector_path = f"data/steering_vector_layer{layer_idx}.npy"

    activations = np.load("data/activations.npy")
    labels = np.load("data/labels.npy")
    steering_vector = np.load(steering_vector_path)
    gold_labels = labels

    scales = [0.1, 0.3, 0.5]

    for scale in scales:
        print(f"\nEvaluating scale: {scale}")

        features_before = prepare_features(activations, layer_idx)
        steered = apply_steering(activations.copy(), layer_idx, scale * steering_vector)
        features_after = prepare_features(steered, layer_idx)

        clf = LogisticRegression(max_iter=1000)
        clf.fit(features_before, gold_labels)

        preds_before = clf.predict(features_before)
        preds_after = clf.predict(features_after)

        print(f"\nClassification report before steering (scale {scale}):")
        print(classification_report(gold_labels, preds_before))

        print(f"\nClassification report after steering (scale {scale}):")
        print(classification_report(gold_labels, preds_after))

        plot_distribution(
            preds_before,
            preds_after,
            save_path=f"plots/prediction_shift_scale_{scale}.png"
        )