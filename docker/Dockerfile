# Reference API image: real biometric matching via ONNXRuntime.
# A 487 KB ONNX export of the trained toy CNN (see
# scripts/export_encoder_onnx.py) replaces the byte-only hash encoder;
# torch stays out, rPPG/flash PAD tiers need only numpy/opencv/faiss.
FROM python:3.10-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

COPY docker/requirements-light.txt /tmp/requirements-light.txt
RUN pip install --no-cache-dir -r /tmp/requirements-light.txt

WORKDIR /app
COPY pyproject.toml README.md ./
COPY abisgpt ./abisgpt
COPY configs ./configs
COPY artifacts/encoder/face_encoder.onnx /app/artifacts/encoder/face_encoder.onnx

RUN pip install --no-cache-dir -e . --no-deps

EXPOSE 8000
CMD ["uvicorn", "abisgpt.api.service:app", "--host", "0.0.0.0", "--port", "8000"]
