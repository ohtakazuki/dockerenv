FROM tensorflow/tensorflow:2.12.0-gpu

# パッケージのインストール
RUN apt-get update -qq \
 && apt-get install -y --no-install-recommends \
    tzdata sudo\
 && rm -rf /var/lib/apt/lists/*

# モジュールのインストール
WORKDIR /tmp/work
COPY requirements.txt ${PWD}
RUN pip install -U pip \
 && pip install -r requirements.txt

# 環境変数の定義
ENV TZ="Asia/Tokyo"

# 一時変数の定義
ARG USERNAME=user1
ARG PASSWORD=user1
ARG GROUPNAME=user1
ARG UID=1000
ARG GID=1000

# 一般権限のユーザーを追加
RUN groupadd -g ${GID} ${GROUPNAME} \
 && useradd -u ${UID} -g ${GID} -G sudo -m -s /bin/bash ${USERNAME} \
 && echo ${USERNAME}:${PASSWORD} | chpasswd \
 && echo "%${USERNAME} ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers

# ユーザーの切替
USER ${USERNAME}
