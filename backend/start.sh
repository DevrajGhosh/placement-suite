#!/bin/bash
echo "Training models..."
cd /opt/render/project/src
python notebooks/02_train_models.py
echo "Starting server..."
gunicorn app:app
