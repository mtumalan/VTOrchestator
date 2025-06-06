# ───────── Dockerfile ─────────

# 1. Base image
FROM python:3.11

# 2. Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      build-essential \
      libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# 3. Install Python dependencies AS ROOT
WORKDIR /usr/src/app
COPY requirements.txt ./
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# 4. Create and switch to a non‐root user
RUN useradd --create-home orchestrator
WORKDIR /home/orchestrator
USER orchestrator

# 5. Copy the rest of the code into that user’s home
COPY --chown=orchestrator:orchestrator . /home/orchestrator/

# 6. Expose port 69 for Django
EXPOSE 8000

# 7. Default entrypoint: run migrations then Gunicorn
CMD ["sh", "-c", "\
        python manage.py migrate && \
        gunicorn orchestrator.wsgi:application \
            --bind 0.0.0.0:8000 \
            --access-logfile - \
            --error-logfile - \
    "]
