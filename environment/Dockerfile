# hash:sha256:d324215abfe073c3e5953657f52536eadabb2844fdadad89b69aa27b3597a978
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

COPY postInstall /
RUN /postInstall
