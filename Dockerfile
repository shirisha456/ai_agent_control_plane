FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1

WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir -e .

COPY alembic.ini ./
COPY migrations ./migrations

# Unprivileged: a worker that can write to its own image is a worker that can
# hide evidence of what it did.
RUN useradd -m acp && chown -R acp /app
USER acp

CMD ["uvicorn", "acp.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
