# Build with this directory as the Docker context after replacing the image
# digest in manifest.json with the digest of the built image.
FROM python:3.11-slim

WORKDIR /models
COPY manifest.json worker.py /models/

# No network, package manager, or model download is performed at runtime.
ENTRYPOINT ["python", "/models/worker.py"]
