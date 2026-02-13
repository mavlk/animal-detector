FROM python:3.12-slim AS base

# System dependencies for OpenCV
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install Poetry
RUN pip install --no-cache-dir poetry==2.1.1 \
    && poetry config virtualenvs.create false

WORKDIR /app

# Install Python dependencies (cached layer)
COPY pyproject.toml poetry.lock ./
RUN poetry install --no-root --only main

# Copy project source, weights, and metadata
COPY README.md ./
COPY src/ src/
COPY weights/ weights/

# Install the project itself
RUN poetry install --only main

ENTRYPOINT ["python", "-m", "animal_detector.batch_report"]
