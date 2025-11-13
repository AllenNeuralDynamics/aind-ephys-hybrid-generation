# hash:sha256:a5bf3755ddb4884ac896510b437cf28827708bf0d3c0283035159bc943f9b0f2
ARG REGISTRY_HOST
FROM $REGISTRY_HOST/codeocean/jupyterlab:3.6.1-miniconda4.12.0-python3.9-ubuntu20.04

ARG DEBIAN_FRONTEND=noninteractive
ARG GIT_ASKPASS
COPY git-askpass /

RUN pip install -U --no-cache-dir \
    s3fs==2024.6.1 \
    spikeinterface[full,widgets]==0.103.0 \
    torch==2.6.0 \
    wavpack-numcodecs==0.2.2
