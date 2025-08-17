# Trained AI/GenAI Models

This directory contains trained AI and generative AI models used in our PBL4 project.

## Overview

The models in this directory are used for various tasks including:
- Natural language processing
- Computer vision
- Predictive analytics
- Text generation

## Model Structure

Each model is stored in its own subdirectory with the following structure:
- `model_name/`: Contains the trained model files
- `model_name/checkpoint/`: Checkpoint files for resuming training
- `model_name/config/`: Configuration files
- `model_name/examples/`: Example inputs and outputs

## Usage

```python
# Example code for loading and using models
from project_utils import load_model

model = load_model('model_name')
result = model.predict(input_data)
```

## Adding New Models

When adding a new model to this directory, please include:
1. The trained model files
2. A README with model specifications
3. Training and evaluation metrics
4. Example usage code

## License Information

Please ensure all models comply with their respective licenses and terms of use.