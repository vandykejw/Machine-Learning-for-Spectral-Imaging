FROM mcr.microsoft.com/devcontainers/python:1-3.12-bullseye

COPY requirements.txt .
RUN apt-get update \
    && apt-get install -y python3-pyqt5 \
    && python -m pip install --no-cache-dir -v -r requirements.txt