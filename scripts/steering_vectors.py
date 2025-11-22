import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from model.steering_utils import compute_steering_vector

if __name__ == "__main__":
    compute_steering_vector(
        hidden_states_file="data/activations.npy",
        labels_file="data/labels.npy",
        target_label=1,           
        contrast_label=0,         
        target_layer=10,          
        save_path="data/steering_vector_layer10.npy"
    )
