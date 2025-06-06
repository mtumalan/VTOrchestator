# Orchestrator

A Django-based service for orchestrating computer vision model inference jobs using Celery and Redis. This project is containerized with Docker and provides REST API endpoints for inference job management.

## Features
- Django 5.2 backend with REST API (using Django REST Framework)
- Asynchronous job processing with Celery and Redis
- Containerized with Docker and Docker Compose
- Gunicorn for production-ready serving

## Requirements
- Docker & Docker Compose
- Python 3.11 (if running locally)

## Setup (with Docker)
1. Clone this repository:
   ```sh
   git clone <repo-url>
   cd orchestrator
   ```
2. Create a `.env` file with the following variables:
   ```env
   VT_API_URL=<your_vt_api_url>
   VT_API_TOKEN=<your_vt_api_token>
   DJANGO_SECRET_KEY=<your_secret_key>
   DJANGO_DEBUG=1
   ```
3. Build and start the services:
   ```sh
   docker-compose up --build
   ```
   This will start:
   - Django web server (on port 69)
   - Celery worker
   - Redis

4. The API will be available at `http://localhost:69/`
   ```

## API Usage
- **POST /enqueue/**
  - Enqueue a new vision model job.
  - Example payload:
    ```json
    {
      "job_id": "<uuid>",
      "vision_model_id": 1,
      "input_image_url": "http://.../file.png"
    }
    ```

## Project Structure
- `core/` – Django app with tasks, views, and business logic
- `orchestrator/` – Django project settings and configuration
- `Dockerfile`, `docker-compose.yml` – Containerization
- `requirements.txt` – Python dependencies
