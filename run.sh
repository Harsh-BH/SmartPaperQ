#!/bin/bash

echo "Setting up Paper-shaper..."

# Ensure script is executable
chmod +x "$0"

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
source venv/bin/activate

# Install requirements and setup NLTK
python setup.py

# Run the application
echo "Starting Paper-shaper application..."
streamlit run app.py

# Deactivate virtual environment when the application closes
deactivate
