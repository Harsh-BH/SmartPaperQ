#!/bin/bash

# Create required directories if they don't exist
mkdir -p papers
mkdir -p papers/arxiv
mkdir -p vectorstore

# Activate virtual environment if it exists
if [ -d "venv" ]; then
  echo "Activating virtual environment..."
  source venv/bin/activate
fi

# Run Streamlit app
echo "Starting SmartPaperQ..."
streamlit run app.py "$@"
