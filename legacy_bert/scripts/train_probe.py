import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import os


if __name__ == "__main__":
    layer_activations = np.load("data/activations.npy")
    emotion_labels = np.load("data/labels.npy")


    num_of_layers = layer_activations.shape[1]
    layer_accuracies = []

    print(f"training layerwise probes on {layer_activations.shape[0]} examples across {num_of_layers} layers")

    for layer in range(num_of_layers):
        activations_for_layer = layer_activations[:, layer, :]


        train_activations, test_activations, train_labels, test_labels = train_test_split(
            activations_for_layer, emotion_labels, test_size=0.2, random_state=42
        )

        classifier = LogisticRegression(max_iter=1000)
        classifier.fit(train_activations, train_labels)
        predictions = classifier.predict(test_activations)

        accuracy = accuracy_score(test_labels, predictions)
        layer_accuracies.append(accuracy)
        print(f"Layer {layer:02d}: Accuracy = {accuracy:.4f}")

    os.makedirs("plots", exist_ok=True)
    plt.figure(figsize=(8, 5))
    plt.plot(range(num_of_layers), layer_accuracies, marker="o")
    plt.title("Layer-wise Probe Accuracy")
    plt.xlabel("BERT Layer")
    plt.ylabel("Classification Accuracy")
    plt.grid(True)
    plt.savefig("plots/layer_probe_accuracy.png")
    print("\n Saved to plots/layer_probe_accuracy.png")
