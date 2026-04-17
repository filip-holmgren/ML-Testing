.PHONY: all data train test clean

# Run everything
all: clean data train test

# Extract / build dataset
data:
	python tools/get_data.py

# Train model
train:
	python src/main.py

# Run tests
test:
	pytest tests/

# Clean generated artifacts
clean:
	rm -rf data/
	rm -rf model/