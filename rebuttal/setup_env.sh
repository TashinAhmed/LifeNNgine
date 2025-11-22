#!/bin/bash

# Change to the directory where the script is located
cd "$(dirname "$0")"

# Create a virtual environment named 'venv_rebuttal'
python3 -m venv venv_rebuttal

# Activate the virtual environment
source venv_rebuttal/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install requirements
pip install -r requirements.txt

# Register the kernel for Jupyter
python -m ipykernel install --user --name=venv_rebuttal --display-name "Python (Rebuttal)"

echo "Environment setup complete. To activate, run: source rebuttal/venv_rebuttal/bin/activate"
