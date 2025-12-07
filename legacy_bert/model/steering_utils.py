import numpy as np


def compute_steering_vector(hidden_states_file: str, labels_file: str, target_label: int, contrast_label: int, target_layer: int, save_path: str) -> None:
    """
    computes and saves a steering vector as the difference between the mean activations for target class vs the contrast class at a given layer
    """
    
    hidden_states = np.load(hidden_states_file)
    labels = np.load(labels_file)

    target_acts = hidden_states[np.array(labels) == target_label, target_layer, :]
    contrast_acts = hidden_states[np.array(labels) == contrast_label, target_layer, :]

    steering_vector = target_acts.mean(axis=0) - contrast_acts.mean(axis=0)
    np.save(save_path, steering_vector)
    print(f"saved steering vector to: {save_path}")


def apply_steering(activations: np.ndarray, layer, steering_vector: np.ndarray) -> np.ndarray:
    layer = int(layer)
    steered = activations.copy()
    steered[:, layer, :] += steering_vector
    return steered