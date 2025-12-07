from typing import List
import torch
import numpy as np
from transformers import AutoModel, AutoTokenizer


class ActivationExtractor:
    def __init__(self, model_name: str = "bert-base-uncased", device: str = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name, output_hidden_states=True)
        self.model.to(self.device)
        self.model.eval()

    def extract_mean_hidden_states(self, texts: List[str], aggregation: str = "mean") -> List[np.ndarray]:
        """
        extract mean hidden state vectors per layer for each input text
        then return a list of [num_layers * hidden_dim] matrices (one per text)
        """

        results = []
        for text in texts:
            inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True).to(self.device)

            with torch.no_grad():
                outputs = self.model(**inputs)
                
                
            hidden_states = outputs.hidden_states

            layer_hidden_states = hidden_states[1:]

            if aggregation == "mean":
                per_layer = [hs.mean(dim=1).squeeze(0).cpu().numpy() for hs in hidden_states]
            elif aggregation == "last":
                per_layer = [hs[:, -1, :].squeeze(0).cpu().numpy() for hs in hidden_states]
            else:
                raise ValueError(f"Unsupported aggregation: {aggregation}")

            results.append(np.stack(per_layer))

        return np.stack(results)

