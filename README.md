# Gemma English Abliterated

A modular Python codebase for reducing refusal behavior in Gemma models through weight orthogonalization. This project implements a research methodology to modify language models to be less likely to refuse requests while maintaining helpfulness and safety.

## Project Structure

```
gemma-english-abliterated/
├── src/                          # Core modules
│   ├── __init__.py              # Package initialization
│   ├── config.py                # Configuration constants
│   ├── utils.py                 # Utility functions
│   ├── model_manager.py         # Model initialization and management
│   ├── orthogonalization.py     # Weight orthogonalization logic
│   ├── evaluation.py            # Benchmarking and evaluation
│   └── model_saver.py           # Model saving utilities
├── run_experiment.py            # Main experiment runner
├── save_model.py                # Save abliterated model
├── use_model.py                 # Load and use saved model
├── requirements.txt             # Python dependencies
└── README.md                    # This file
```

## Features

- **Modular Design**: Clean separation of concerns with dedicated modules
- **Comprehensive Evaluation**: Multiple metrics for measuring refusal behavior
- **Flexible Configuration**: Centralized configuration management
- **Robust Error Handling**: Comprehensive error handling and logging
- **Multiple Output Formats**: Save models in HuggingFace format and Ollama-compatible format

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd gemma-english-abliterated
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Running the Complete Experiment

To run the full ablation experiment (baseline → orthogonalization → evaluation):

```bash
python run_experiment.py
```

This will:
1. Load the Gemma-2B-IT model
2. Run baseline benchmarks
3. Apply weight orthogonalization
4. Run post-modification benchmarks
5. Compare results

### Saving the Modified Model

To save the abliterated model for later use:

```bash
python save_model.py
```

This creates:
- HuggingFace format model files
- Ollama Modelfile
- Abliterator cache file
- Model information file

### Using the Saved Model

To load and use the saved abliterated model:

```bash
python use_model.py
```

This provides:
- Automated testing with various prompts
- Interactive chat mode
- Model response analysis

## Configuration

All configuration is centralized in `src/config.py`:

- **Model Settings**: Model ID, chat template, activation layers
- **Generation Parameters**: Token limits, batch sizes, sampling parameters
- **Token Sets**: Positive/negative tokens for scoring
- **Refusal Detection**: Phrases used to detect refusals

## Modules

### `src/config.py`
Centralizes all configuration constants and settings.

### `src/utils.py`
Common utility functions for tokenization, response handling, and validation.

### `src/model_manager.py`
Handles model initialization, configuration, and lifecycle management.

### `src/orthogonalization.py`
Core logic for computing and applying refusal direction orthogonalization.

### `src/evaluation.py`
Benchmarking and evaluation functions for measuring model performance.

### `src/model_saver.py`
Utilities for saving models in different formats (HuggingFace, Ollama).

## Results

The orthogonalization technique typically achieves:
- **Harmful Prompts**: ~6% reduction in refusal scores
- **Harmless Prompts**: ~43% reduction in refusal scores

## Research Context

This project implements weight orthogonalization as described in research on reducing model refusal behavior. The technique computes refusal directions from activation differences between harmful and harmless prompts, then orthogonalizes model weights to reduce the influence of these directions.

## Safety Considerations

- This research is for legitimate safety research purposes
- The modified model should be used responsibly
- Monitor outputs for potential misuse
- Consider ethical implications of reduced refusal behavior

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

[Add your license information here]

## Citation

If you use this code in your research, please cite:

```bibtex
[Add citation information here]
```
