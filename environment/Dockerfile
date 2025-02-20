# hash:sha256:965c68e3c8013a984454249d2d15233e0006a51e4bca6b11c99e99354f1ead1c
ARG REGISTRY_HOST
FROM $REGISTRY_HOST/codeocean/jupyterlab:3.6.1-miniconda4.12.0-python3.9-ubuntu20.04

ARG DEBIAN_FRONTEND=noninteractive

ARG GIT_ASKPASS
ARG GIT_ACCESS_TOKEN
COPY git-askpass /

RUN pip install -U --no-cache-dir \
    s3fs==2024.6.1 \
    spikeinterface[full,widgets]==0.102.1 \
    torch==2.6.0 \
    wavpack-numcodecs==0.2.2

COPY postInstall /
RUN /postInstall
