# hash:sha256:a211d732eb06391bf8f10e4545ee42c9525d0ae21e117b00eb112177f19ca0aa
ARG REGISTRY_HOST
FROM $REGISTRY_HOST/codeocean/jupyterlab:3.6.1-miniconda4.12.0-python3.9-ubuntu20.04

ARG DEBIAN_FRONTEND=noninteractive

ARG GIT_ASKPASS
ARG GIT_ACCESS_TOKEN
COPY git-askpass /

RUN pip install -U --no-cache-dir \
    s3fs==2024.6.1 \
    spikeinterface[full,widgets]==0.101.0 \
    wavpack-numcodecs==0.1.5
