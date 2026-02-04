# Predictive maintenance for IOT Sensors with Docker
A machine learning project that predicts sensor failures using XGBoost, containerized with Docker for easy deployment and reproducibility.

## Project Overview
This project demonstrates:

Building a machine learning model to predict machine failures

Creating production-ready Docker containers

Orchestrating services with Docker Compose

Publishing images to Docker Hub

Model training and inference pipelines

## Features
Predictive Maintenance: XGBoost model trained on sensor data

Dockerized: Complete containerization of the application

Multi-service Architecture: Separate services for training and inference

Easy Deployment: One-command setup with Docker Compose

Scalable: Designed for production environments

## Architecture

Training Service: Handles model training and evaluation


## Project Structure
```
project-root/
│
├── Dockerfile              # Dockerfile for training service
├── docker-compose.yml      # Service orchestration
├── requirements.txt        # Python dependencies
├── initial_script.ipynb    # Initial exploratory notebook
│
├── data/                   # Data directory
│   ├── raw/               # Raw data files
│   └── processed/         # Processed data
│       ├── X.parquet
│       └── y.parquet
│
├── models/                 # Trained model files
│
└── src/                    # Source code
    ├── __init__.py
    ├── model.py           # Model training script
    └── preprocessing.py   # Data preprocessing
```
## Prerequisites
Docker Engine 20.10+

Docker Compose 2.0+

Python 3.8+ (for local development)

Git

Quick Start with Docker
1. Clone the Repository
bash
git clone <repository-url>
cd machine-failure-prediction
2. Build and Run with Docker Compose
bash
# Build and start all services
docker-compose up --build

# Run in detached mode
docker-compose up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down
