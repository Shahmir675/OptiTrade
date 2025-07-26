FROM python:3.13-slim AS builder

WORKDIR /app

RUN pip install --upgrade pip && pip install uv

COPY requirements.txt .

# Install dependencies first (takes advantage of Docker cache)
RUN uv pip install --system -r requirements.txt

# Now copy the rest of the app
COPY . .

# Optional: compile to bytecode (comment if unnecessary)
RUN python -m compileall -b .

RUN find . -name "*.py" -type f -delete

# -------- Stage 2: Final image --------
FROM python:3.13-slim

WORKDIR /app

# Install uv only (for runtime)
RUN pip install uv

RUN apt-get update && apt-get install -y curl

# Copy from builder
COPY --from=builder /app /app
COPY --from=builder /usr/local/lib/python3.13/site-packages /usr/local/lib/python3.13/site-packages

ENV PYTHONUNBUFFERED=1
EXPOSE 8001

CMD ["uv", "run", "/app/run.pyc"]
