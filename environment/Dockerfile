# hash:sha256:fe7723252d28bc800848c9d8a368b538cc7eb896855a6a49d23e2cf0c4aad871
ARG REGISTRY_HOST
FROM $REGISTRY_HOST/codeocean/jupyterlab:3.6.1-miniconda4.12.0-python3.9-ubuntu20.04

ARG DEBIAN_FRONTEND=noninteractive

ARG GIT_ASKPASS
ARG GIT_ACCESS_TOKEN
COPY git-askpass /

RUN pip install -U --no-cache-dir \
    s3fs==2024.6.1 \
    spikeinterface[full,widgets]==0.102.3 \
    torch==2.6.0 \
    wavpack-numcodecs==0.2.2

COPY postInstall /
RUN /postInstall
