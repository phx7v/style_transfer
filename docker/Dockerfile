# -----------------------------
# 1) Builder stage
# -----------------------------
FROM python:3.11-slim AS builder

LABEL authors="phx7v"

ENV DEBIAN_FRONTEND=noninteractive

WORKDIR /style_transfer

ENV POETRY_VERSION=2.3.1

RUN pip install --upgrade pip && pip install --no-cache-dir poetry==$POETRY_VERSION

# Disable venv creation (install deps globally)
RUN poetry config virtualenvs.create false

COPY pyproject.toml poetry.lock ./

# Install runtime dependencies
RUN poetry install --no-root --only main

# -----------------------------
# 2) Dev stage (with dev deps)
# -----------------------------
FROM builder AS dev

# Install dev dependencies on top of builder
RUN poetry install --no-root

# Copy project code
COPY . .

EXPOSE 8501
ENV PYTHONPATH=/style_transfer

CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.headless=true"]

# -----------------------------
# 3) Production stage
# -----------------------------
FROM python:3.11-slim AS prod

LABEL authors="phx7v"

ENV DEBIAN_FRONTEND=noninteractive

WORKDIR /style_transfer

# Copy only installed dependencies from builder
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy project code
COPY . .

EXPOSE 8501
ENV PYTHONPATH=/style_transfer

CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.headless=true"]
