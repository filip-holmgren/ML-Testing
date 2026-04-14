# ML Testing

## Running
The project uses `uv` for python venv management

To run the project simply run
```bash
$ uv run ./main.py
```

## Current results
```
Final Test Accuracy: 0.9794

Classification Report:
              precision    recall  f1-score   support

         0.0       0.99      0.99      0.99       270
         1.0       0.89      0.81      0.85        21

    accuracy                           0.98       291
   macro avg       0.94      0.90      0.92       291
weighted avg       0.98      0.98      0.98       291
```

Class 0.0 is true positive
Class 1.0 is false positive