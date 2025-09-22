# Quantum-Enhanced Precision Agriculture - Makefile

.PHONY: help install test demo build run clean

# Default target
help:
	@echo "Quantum-Enhanced Precision Agriculture - Available Commands:"
	@echo ""
	@echo "  make install    - Install dependencies"
	@echo "  make test       - Run unit tests"
	@echo "  make demo       - Run interactive demo"
	@echo "  make build      - Build Docker image"
	@echo "  make run        - Run main pipeline"
	@echo "  make clean      - Clean output files"
	@echo "  make help       - Show this help message"
	@echo ""
	@echo "Before running, update config/config.py with your GCP project details!"

# Install dependencies
install:
	@echo "Installing dependencies..."
	pip install -r requirements.txt
	@echo "Dependencies installed"

# Run unit tests
test:
	@echo "Running unit tests..."
	python -m pytest tests/ -v
	@echo " Tests completed"

# Run interactive demo
demo:
	@echo "Running interactive demo..."
	python demo_agriculture.py
	@echo " Demo completed"

# Run main pipeline
run:
	@echo "Running main pipeline..."
	python main.py
	@echo "Pipeline completed"

# Build Docker image
build:
	@echo "Building Docker image..."
	docker build -t aiquery:latest .
	@echo "Docker image built: aiquery:latest"

# Run with Docker
docker-run:
	@echo "Running with Docker..."
	docker run --rm -v $(PWD)/outputs:/app/outputs aiquery:latest
	@echo "Docker run completed"

# Clean output files
clean:
	@echo "Cleaning output files..."
	rm -rf outputs/*.csv
	rm -rf __pycache__/
	rm -rf src/__pycache__/
	rm -rf config/__pycache__/
	@echo " Clean completed"

# Development setup
dev-setup: install
	@echo "Setting up development environment..."
	mkdir -p outputs
	@echo "Development environment ready"
	@echo "Next steps:"
	@echo "1. Update config/config.py with your GCP project details"
	@echo "2. Run 'make demo' to test the setup"

# Full test suite
test-all: test demo
	@echo "All tests completed"

# Production build
prod-build: clean build
	@echo "Production build completed"
