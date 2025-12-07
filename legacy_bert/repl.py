import code
import sys
import os

sys.path.append(os.path.abspath("."))

import numpy as np
import torch
from pprint import pprint

from model.activation_utils import ActivationExtractor
from model.steering_utils import compute_steering_vector
from data.goemotions_loader import load_goemotions


def show_docs():
    print("Welcome to the Emotion Steering REPL!")
    print("\nUseful modules:")
    print(" - load_goemotions() from data.goemotions_loader")
    print(" - ActivationExtractor from model.activation_utils")
    print(" - compute_steering_vector from model.steering_utils")
    print(" - activations = np.load('data/activations.npy')")
    print(" - labels = np.load('data/labels.npy')")
    print("\nExamples:")
    print(">>> texts, labels = load_goemotions('joy', 'neutral', max_samples=10)")
    print(">>> extractor = ActivationExtractor()")
    print(">>> states = extractor.extract_mean_hidden_states(texts)")
    print(">>> print(states[0].shape)")
    print(">>> steer_layer(10)")

def inspect_data():
    activations = np.load("data/activations.npy")
    labels = np.load("data/labels.npy")
    print("Activations:", activations.shape)
    print("Labels:", labels.shape)
    print("Example label:", labels[0])
    print("Example activation shape:", activations[0].shape)

def steer_layer(layer=10):
    activations = np.load("data/activations.npy")
    vec_path = f"data/steering_vector_layer{layer}.npy"
    if not os.path.exists(vec_path):
        print(f"Steering vector for layer {layer} not found at: {vec_path}")
        return None
    steering_vector = np.load(vec_path)

    steered = activations.copy()
    steered[:, layer, :] += steering_vector

    before = activations[:, layer, :].mean()
    after = steered[:, layer, :].mean()

    print(f"Steered layer {layer}")
    print(f"- Mean before steer: {before:.4f}")
    print(f"- Mean after  steer: {after:.4f}")
    return steered


vars = globals().copy()
vars.update(locals())

vars["show_docs"] = show_docs
vars["inspect_data"] = inspect_data
vars["steer_layer"] = steer_layer

show_docs()
code.interact(local=vars)
