FROM public.ecr.aws/docker/library/python:3.11.8-bullseye as be

# Install system dependencies
RUN apt-get update && apt-get install -y \
    jq \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Set up virtual environment
RUN python -m venv /bot-venv
ENV VIRTUAL_ENV=/bot-venv
ENV PATH="/bot-venv/bin:$PATH"

# Set working directory
WORKDIR /app

# Upgrade pip
RUN pip install --upgrade pip
# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Set Python path
ENV PYTHONPATH /app
