# ベースイメージ: slim イメージを使用
FROM python:3.13-slim 

# タイムゾーン設定と必要なOSパッケージのインストール
RUN apt-get update -qq && \
    apt-get install -y --no-install-recommends tzdata locales && \
    ln -sf /usr/share/zoneinfo/Asia/Tokyo /etc/localtime && \
    echo "Asia/Tokyo" > /etc/timezone && \
    echo "ja_JP.UTF-8 UTF-8" >> /etc/locale.gen && \
    locale-gen ja_JP.UTF-8 && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# 環境変数: 基本設定
ENV TZ=Asia/Tokyo \
    LANG=ja_JP.UTF-8 \
    LANGUAGE=ja_JP:ja \
    LC_ALL=ja_JP.UTF-8 \
    PYTHONUNBUFFERED=1 \
    NB_USER=jovyan \
    NB_UID=1000 \
    NB_GID=100

# 環境変数: NB_USER を使って HOME を定義
ENV HOME=/home/${NB_USER}

# 環境変数: HOME を使って APP_DIR を定義
ENV APP_DIR=${HOME}/work

# 環境変数: HOME と既存の PATH を使って PATH を定義
ENV PATH="${HOME}/.local/bin:${PATH}"

# 作業ディレクトリの作成、ユーザー作成、権限設定
RUN useradd -r -g ${NB_GID} -u ${NB_UID} -m -d ${HOME} -s /bin/bash ${NB_USER} && \
    mkdir -p ${APP_DIR} && \ 
    chown -R ${NB_UID}:${NB_GID} ${HOME} 

# 依存関係ファイルを先にコピー (権限も設定)
COPY --chown=${NB_USER}:${NB_USER} requirements.txt /tmp/requirements.txt

# 依存関係をインストール
USER ${NB_USER}
WORKDIR ${APP_DIR} 

RUN pip install --no-cache-dir -r /tmp/requirements.txt && \
    rm /tmp/requirements.txt && \
    rm -rf ${HOME}/.cache/pip

# Jupyter Notebook サーバーが使用するポート
EXPOSE 8888

# デフォルトのコンテナ起動コマンド (compose.yml の command で上書きされる)
# CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--notebook-dir=/home/jovyan/work"]