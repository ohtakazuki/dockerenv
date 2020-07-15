FROM python:3.8-slim

# パッケージのインストール
RUN apt-get update && apt-get install -y \
    tzdata \
 && rm -rf /var/lib/apt/lists/*

# モジュールのインストール
WORKDIR /tmp/work
COPY requirements.txt ${PWD}
RUN pip install -r requirements.txt

# 環境変数の定義
ENV TZ=Asia/Tokyo
ENV USER user1

# 一般権限のユーザーを追加
RUN useradd -m ${USER} --uid 1000
RUN gpasswd -a ${USER} sudo
RUN echo "${USER}:password" | chpasswd

# ユーザーの切替
USER ${USER}
