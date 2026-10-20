# OpenABIS reference API image — deliberately torch-free.
# The hash encoder + rPPG/flash PAD tiers need only numpy/opencv/faiss;
# the full torch encoder path is opt-in via the host training workflow.
FROM python:3.10-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

COPY docker/requirements-light.txt /tmp/requirements-light.txt
RUN pip install --no-cache-dir -r /tmp/requirements-light.txt

WORKDIR /app
COPY pyproject.toml README.md ./
COPY openabis ./openabis
COPY configs ./configs

RUN pip install --no-cache-dir -e . --no-deps

EXPOSE 8000
CMD ["uvicorn", "openabis.api.service:app", "--host", "0.0.0.0", "--port", "8000"]
