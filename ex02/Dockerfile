FROM python:3.11-slim

# 作業ディレクトリを変更
WORKDIR /tmp/mydir

# ファイルのコピー
COPY requirements.txt ${PWD}

# pip自体のアップデート
RUN pip install -U pip

# pythonのパッケージをインストール
RUN pip install -r requirements.txt

# 一時変数の定義
ARG wdir

# 作業ディレクトリを変更
WORKDIR $wdir

