# ML Project

This project uses **uv** for Python dependency management and a Makefile to orchestrate the workflow.

## Requirements

- Python 3.10+
- [uv](https://github.com/astral-sh/uv)
- PostgreSQL database access

## Setup

Create environment and install dependencies:
```bash
$ uv sync
```

## Environment Variables
| env var      | description                                          |
|--------------|------------------------------------------------------|
| DATABASE_URL | The url of the postgresql database you want to query |


## Makefile Commands
### Run full pipeline
```bash
$ make all
```

### Extract dataset from database
```bash
make data
```

### Train model
```bash
$ make train
```

### Run tests
```bash
$ make test
```

### Clean generated artifacts
```bash
$ make clean
```

## Testing
Run tests with:
```bash
$ pytest tests/
```

## Current performance
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

> [!NOTE]
> * Class 0.0 is true positive 
> * Class 1.0 is false positive