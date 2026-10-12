# syntax=docker/dockerfile:1

FROM python:3.10.12-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        git \
        libgl1 \
        libgomp1 \
        pkg-config \
        libcairo2-dev \
        libgirepository1.0-dev \
        libglib2.0-dev \
        libffi-dev \
        python3-dev && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

COPY requirements.txt /tmp/requirements.txt

RUN --mount=type=cache,target=/root/.cache/pip \
    python -m pip install --no-deps \
    --extra-index-url https://download.pytorch.org/whl/cu128 \
    -r /tmp/requirements.txt

CMD ["bash"]