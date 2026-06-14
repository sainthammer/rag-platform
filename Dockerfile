FROM python:3.11-slim

WORKDIR /app

# CPU-only torch first — prevents pip from pulling 2GB of NVIDIA CUDA packages
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

COPY pyproject.toml .
RUN pip install --no-cache-dir -e ".[eval]"

# Pre-download tiktoken encodings so the container doesn't need internet at runtime
RUN python -c "import tiktoken; tiktoken.get_encoding('cl100k_base'); tiktoken.get_encoding('o200k_base')"

COPY . .

EXPOSE 8080

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8080"]
