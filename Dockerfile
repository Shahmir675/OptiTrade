FROM python:3.13-slim AS builder

WORKDIR /app

RUN pip install --upgrade pip && pip install uv

COPY requirements.txt .

RUN uv pip install --system -r requirements.txt

COPY . .

RUN python -m compileall -b .

RUN find . -name "*.py" -type f -delete

FROM python:3.13-slim

WORKDIR /app

RUN pip install uv

RUN apt-get update && apt-get install -y curl

COPY --from=builder /app /app
COPY --from=builder /usr/local/lib/python3.13/site-packages /usr/local/lib/python3.13/site-packages

ENV PYTHONUNBUFFERED=1
EXPOSE 8001

CMD ["uv", "run", "/app/run.pyc"]