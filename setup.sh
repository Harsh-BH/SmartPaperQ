#!/bin/bash

echo "Setting up SmartPaperQ environment..."

# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Make run.sh executable
chmod +x run.sh

# Create required directories
mkdir -p papers
mkdir -p papers/arxiv
mkdir -p vectorstore

echo "Setup complete! Run './run.sh' to start the application."
