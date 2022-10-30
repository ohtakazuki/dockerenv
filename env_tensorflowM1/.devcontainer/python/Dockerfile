FROM sonoisa/deep-learning-coding:pytorch1.12.0_tensorflow2.9.1

# パッケージのインストール
RUN sudo apt-get update -qq \
 && sudo apt-get install -y --no-install-recommends \
    tzdata\
 && sudo rm -rf /var/lib/apt/lists/*

# モジュールのインストール
SHELL ["/bin/bash", "-c"]
WORKDIR /home/ubuntu
COPY requirements.txt ${PWD}
RUN source python3-venv/bin/activate \
 && pip install -U pip \
 && pip install -r requirements.txt

# 環境変数の定義
ENV TZ="Asia/Tokyo"
