#!/usr/bin/env/bash

# Create virtual environment 
python3 -m venv assignment_4

# Activate virtual environment 
source ./assignment_4/bin/activate

#Installing requirements
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt

# Deactivate the virtual environment 
deactivate