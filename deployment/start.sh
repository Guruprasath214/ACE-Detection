#!/bin/bash

# ACE Detection Framework Startup Script

set -e

echo "Starting ACE Detection Framework..."

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo "Error: Docker is not installed. Please install Docker first."
    exit 1
fi

# Check if Docker Compose is installed
if ! command -v docker-compose &> /dev/null; then
    echo "Error: Docker Compose is not installed. Please install Docker Compose first."
    exit 1
fi

# Create necessary directories
echo "Creating necessary directories..."
mkdir -p models/saved
mkdir -p data/raw
mkdir -p data/processed
mkdir -p logs
mkdir -p assets

# Check if models exist
if [ ! -f "models/saved/ace_ensemble/meta_classifier.pth" ]; then
    echo "Warning: No trained models found. Please train the models first using:"
    echo "python train/train_pipeline.py --data data/annotated.csv"
    echo ""
    echo "Continuing with default initialization..."
fi

# Build and start services
echo "Building and starting services..."
docker-compose -f deployment/docker-compose.yml up --build -d

# Wait for services to be ready
echo "Waiting for services to be ready..."
sleep 30

# Check service health
echo "Checking service health..."
if curl -f http://localhost:8000/health > /dev/null 2>&1; then
    echo "✅ ACE API is healthy"
else
    echo "❌ ACE API is not responding"
fi

if curl -f http://localhost:3000 > /dev/null 2>&1; then
    echo "✅ Grafana is running"
else
    echo "❌ Grafana is not responding"
fi

if curl -f http://localhost:9090 > /dev/null 2>&1; then
    echo "✅ Prometheus is running"
else
    echo "❌ Prometheus is not responding"
fi

echo ""
echo "🎉 ACE Detection Framework is now running!"
echo ""
echo "Services:"
echo "  - API Server: http://localhost:8000"
echo "  - API Docs: http://localhost:8000/docs"
echo "  - Grafana: http://localhost:3000 (admin/admin)"
echo "  - Prometheus: http://localhost:9090"
echo ""
echo "To stop the services, run:"
echo "  docker-compose -f deployment/docker-compose.yml down"
echo ""
echo "To view logs, run:"
echo "  docker-compose -f deployment/docker-compose.yml logs -f"
