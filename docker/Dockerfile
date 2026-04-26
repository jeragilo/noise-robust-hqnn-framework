FROM python:3.11-slim

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential git nano && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# Copy requirements first (for layer caching)
COPY env/requirements.txt /tmp/requirements.txt

RUN pip install --no-cache-dir -r /tmp/requirements.txt

CMD ["bash"]
