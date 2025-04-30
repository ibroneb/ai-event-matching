# README.md content

"""
#AI Event Matching System

This project demonstrates a basic AI-based event recommendation engine using TensorFlow. It uses a neural network with embeddings to learn match scores between users and campus events based on historical interaction data.

>Note: This sample implementation is inspired by a proprietary matching system I developed during a previous internship. All data and code here are synthetic and for demonstration purposes only.

## Features
- User and event embeddings
- Dot-product similarity for recommendations
- TensorFlow-based model architecture
- Minimal dummy dataset for demonstration

## Technologies Used
- Python
- TensorFlow / Keras
- Pandas
- StringLookup for categorical encoding

## Getting Started
1. Clone this repo
2. Install dependencies:
   ```bash
   pip install tensorflow pandas
   ```
3. Run the script:
   ```bash
   python ai_matching_model.py
   ```

## Example Output
```
Epoch 1/10 ...
Predicted match score: 0.8234
```

## Future Improvements
- Replace dummy data with real event/user metadata
- Use softmax or ranking loss
- Integrate with a mobile frontend or chatbot
- Implement Top-N recommendation system

## Author
Ibrahim Neberai — AI/ML, SWE
