.PHONY: all data train test clean

PY = uv run python
PYTEST = uv run pytest

# Run everything
all: clean data train test

# Extract / build dataset
data:
	$(PY) tools/get_data.py $(ARGS)

# Train model
train:
	$(PY) -m src.main $(ARGS)

# Run tests
test:
	$(PYTEST) $(ARGS)

# Clean generated artifacts
clean:
	rm -rf data/
	rm -rf model/