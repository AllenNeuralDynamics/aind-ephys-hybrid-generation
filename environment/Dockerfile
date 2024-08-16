# hash:sha256:b1c242250cf319b12119bbaf16a95228e64a23ef76e2154fd0a658d95bbf0748
ARG REGISTRY_HOST
FROM $REGISTRY_HOST/codeocean/jupyterlab:3.6.1-miniconda4.12.0-python3.9-ubuntu20.04

ARG DEBIAN_FRONTEND=noninteractive

ARG GIT_ASKPASS
ARG GIT_ACCESS_TOKEN
COPY git-askpass /

RUN pip install -U --no-cache-dir \
    s3fs==2024.6.1 \
    spikeinterface[full,widgets]==0.101.0 \
    torch==2.2.0 \
    wavpack-numcodecs==0.1.5
