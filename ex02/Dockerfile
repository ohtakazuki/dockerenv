FROM python:3.13-slim

# 作業ディレクトリを変更
WORKDIR /tmp/mydir

# ファイルのコピー
COPY requirements.txt .

# pip自体のアップデート
# --no-cache-dir: pipのキャッシュを使わない (イメージサイズ削減)
RUN pip install --no-cache-dir -U pip

# pythonのパッケージをインストール
RUN pip install --no-cache-dir -r requirements.txt

# 一時変数の定義
ARG wdir

# 作業ディレクトリを変更
WORKDIR $wdir