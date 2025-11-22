
# Emotion Steering: Model Self-Representation vs User Attribution

[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.9+-orange.svg)](https://pytorch.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A research project investigating whether activation steering affects a language model's self-representation or its perception of user emotions. This work explores the fundamental mechanisms behind persona-induced misalignment in AI systems.

## Research Question

When we steer a language model toward an emotional state (e.g., "joy"), does the model adopt that emotion itself, or does it attribute that emotion to the user? This distinction is critical for understanding AI safety, persona dynamics, and controllable language model design.

## Key Findings

- **Low-magnitude steering** (α = 0.1) preserves accuracy (96%) while inducing measurable emotional shifts
- **High-magnitude steering** (α = 0.5) produces strong effects but degrades performance (80% accuracy)
- Steering vectors consistently push activations toward target emotion classes
- Layer 10 of DistilBERT shows optimal balance of semantic richness and steering effectiveness

## Quick Start
```bash
# Clone the repository
git clone https://github.com/your-username/EMOTION_STEERING.git
cd EMOTION_STEERING

# Set up virtual environment (if not already exists)
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Extract activations from DistilBERT
python scripts/extract_activations.py

# Train layer-wise probes to find optimal layer
python scripts/train_probe.py

# Create steering vectors
python scripts/steering_vectors.py

# Evaluate steering effects across different magnitudes
python scripts/evaluate_steering.py

# Interactive exploration (optional)
python repl.py
```

## Project Structure
```
EMOTION_STEERING/
├── .venv/                      # Virtual environment
├── .vscode/
│   └── settings.json
├── data/
│   ├── __pycache__/
│   ├── activations.npy         # Extracted hidden states
│   ├── goemotions_loader.py    # Dataset loading utilities
│   ├── labels.npy              # Emotion labels
│   └── steering_vector_layer10.npy  # Computed steering vector
├── model/
│   ├── __pycache__/
│   │   ├── __init__.cpython-313.pyc
│   │   ├── activation_utils.cpython-313.pyc
│   │   └── steering_utils.cpython-313.pyc
│   ├── __init__.py
│   ├── activation_utils.py     # Activation extraction utilities
│   └── steering_utils.py       # Steering vector operations
├── plots/
│   ├── layer_probe_accuracy.png        # Layer-wise accuracy analysis
│   ├── prediction_shift_scale_0.1.png  # Low-magnitude steering effects
│   ├── prediction_shift_scale_0.3.png  # Medium-magnitude steering effects
│   ├── prediction_shift_scale_0.5.png  # High-magnitude steering effects
│   └── prediction_shift.png            # Overall prediction shifts
├── scripts/
│   ├── evaluate_steering.py    # Steering evaluation pipeline
│   ├── extract_activations.py  # Activation extraction script
│   ├── steering_vectors.py     # Steering vector construction
│   └── train_probe.py          # Linear probe training
├── README.md
├── repl.py                     # Interactive testing/exploration
└── requirements.txt
```

## Methodology

### Dataset
- **GoEmotions**: 58k Reddit comments with 28 emotion categories
- **Filtering**: Binary subset of "joy" vs "neutral" (15,671 examples)
- **Class distribution**: 91% neutral, 9% joy (natural distribution)

### Model Architecture
- **Base model**: DistilBERT (66M parameters, 6 layers)
- **Intervention layer**: Layer 10 (optimal semantic/steering balance)
- **Activation extraction**: Mean pooling across token positions
- **Steering vector storage**: Pre-computed in `steering_vector_layer10.npy`

### Steering Vector Construction
```python
v_steer = μ_joy - μ_neutral
A' = A + α × v_steer
```
Where α ∈ {0.1, 0.3, 0.5} represents steering magnitude.

### Evaluation Methods
1. **Classification metrics**: Accuracy, precision, recall, F1-score
2. **Attribution analysis**: Self-representation vs user-representation probes
3. **Layerwise sensitivity**: Linear probe accuracy across all transformer layers

## Results Summary

| Steering Magnitude (α) | Accuracy | Precision (Joy) | Recall (Joy) | F1 (Joy) |
|----------------------|----------|----------------|--------------|----------|
| 0.0 (baseline)       | 96%      | 0.87           | 0.71         | 0.78     |
| 0.1 (low)            | 96%      | 0.78           | 0.81         | 0.79     |
| 0.3 (moderate)       | 92%      | 0.53           | 0.93         | 0.67     |
| 0.5 (high)           | 80%      | 0.31           | 0.98         | 0.47     |

### Key Insights
- **Recall-Precision Tradeoff**: Higher steering magnitude increases emotion detection but reduces specificity
- **Semantic Boundary**: α = 0.5 pushes representations beyond learned manifold, causing performance degradation
- **Optimal Operating Point**: α = 0.1 provides meaningful steering with minimal performance cost
- **Layer Analysis**: Layer 10 provides optimal balance between semantic richness and steering effectiveness

### Visualization Results
The `plots/` directory contains comprehensive visualizations:
- **layer_probe_accuracy.png**: U-shaped accuracy pattern across DistilBERT layers
- **prediction_shift_*.png**: Distributional changes for each steering magnitude
- **prediction_shift.png**: Overall comparison of steering effects

## Current Status & Next Steps

### Completed
- [x] Dataset preprocessing and filtering
- [x] Activation extraction pipeline
- [x] Steering vector construction
- [x] Baseline classification metrics
- [x] Layerwise sensitivity analysis

### In Progress
- [ ] Self-representation vs user-attribution probe evaluation
- [ ] Quantitative attribution metrics development
- [ ] Statistical significance testing

### Future Work
- [ ] Extension to additional emotion categories (fear, anger, anxiety)
- [ ] Evaluation on larger models (GPT, LLaMA architectures)
- [ ] Generative task evaluation beyond classification
- [ ] Real-world deployment safety analysis

## Installation & Dependencies
```bash
pip install torch transformers datasets scikit-learn matplotlib seaborn pandas numpy
```

See `requirements.txt` for specific versions.
## Citation

If you use this work in your research, please cite:
```bibtex
@article{ajayi2025emotion,
  title={Distinguishing Model Self-Representation from User Attribution in Emotion Steering},
  author={Ajayi, Elena},
  journal={Supervised Program for Alignment Research},
  year={2025}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Supervised Program for Alignment Research (SPAR) - Fall 2025
- GoEmotions dataset authors
- Hugging Face transformers library
- DistilBERT architecture developers

## Contact

Elena Ajayi - elenaajayi@outlook.com
