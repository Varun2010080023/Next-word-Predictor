# Next Word Predictor using LSTM and RNN

## Overview
This project implements a Next Word Predictor using a Recurrent Neural Network (RNN) with Long Short-Term Memory (LSTM) layers. The model is trained on Shakespeare's *Hamlet* text data to predict the next word in a given sequence of words.

## Dataset
The dataset used for training is based on the text from *Hamlet*, one of Shakespeare's most famous plays. The text is preprocessed by:
- Removing special characters and unnecessary whitespace
- Converting to lowercase
- Tokenizing into words and sequences

## Model Architecture
The model is built using TensorFlow and Keras, consisting of the following layers:
1. **Embedding Layer** - Converts words into dense vector representations
2. **LSTM Layers** - Captures long-term dependencies in the text
3. **Dense Layer** - Outputs probabilities for the next word prediction

## Installation
To run this project, install the required dependencies:
```bash
pip install tensorflow numpy pandas nltk
```

## Usage
Run the script to train the model:
```bash
python train.py
```
After training, use the model for predictions:
```python
from predictor import predict_next_word

text = "To be or not to"
next_word = predict_next_word(text)
print("Predicted next word:", next_word)
```

## Performance
- The model's accuracy improves with more training epochs and a larger dataset.
- Since Shakespearean language is complex, results may vary depending on the training data's diversity.

## Future Enhancements
- Implementing a Transformer-based model for better contextual understanding
- Fine-tuning the model on additional Shakespearean plays
- Expanding the dataset for broader text generalization



