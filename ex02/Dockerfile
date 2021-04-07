FROM python:3.9-slim

# 作業ディレクトリを変更
WORKDIR /tmp/mydir

# ファイルのコピー
COPY requirements.txt ${PWD}

# pythonのパッケージをインストール
RUN pip install -r requirements.txt

# 一時変数の定義
ARG wdir

# 作業ディレクトリを変更
WORKDIR $wdir

