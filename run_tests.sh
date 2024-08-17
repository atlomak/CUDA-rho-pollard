#!/bin/bash

# Check if the 'sage' command is available
if command -v sage >/dev/null 2>&1; then
    echo "SageMath is installed."
    sage --version
    
    sage -python -m pytest -s
else
    echo "SageMath is not installed or not in the PATH."
fi
