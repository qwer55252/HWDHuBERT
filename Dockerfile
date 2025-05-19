FROM pytorch/pytorch:2.5.1-cuda11.8-cudnn9-runtime

# NVIDIA Container Toolkit 설치 (컨테이너내 설치 시 호스트 권한 필요)
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        ca-certificates \
        curl \
        gnupg \
        git \
        lsb-release && \
    curl -fsSL https://nvidia.github.io/nvidia-docker/gpgkey | apt-key add - && \
    distribution=$(. /etc/os-release; echo $ID$VERSION_ID) && \
    curl -fsSL https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list \
        | tee /etc/apt/sources.list.d/nvidia-docker.list && \
    apt-get update && \
    apt-get install -y --no-install-recommends nvidia-docker2 && \
    rm -rf /var/lib/apt/lists/*

# 필수 Python 패키지 설치
RUN pip install \
    torch==2.5.1+cu118 \
    torchvision==0.20.1+cu118 \
    torchaudio==2.5.1+cu118 \
    pytorch-lightning==2.5.1 \
    open-clip-torch==2.24.0 \
    torch-complex==0.4.4 \
    torch-optimizer==0.3.0 \
    torch-stoi==0.2.3 \
    torchdiffeq==0.2.5 \
    torchinfo==1.8.0 \
    torchmetrics==1.7.1 \
    torchprofile==0.0.4 \
    torchsummaryX==1.3.0

WORKDIR /workspace
COPY . /workspace
