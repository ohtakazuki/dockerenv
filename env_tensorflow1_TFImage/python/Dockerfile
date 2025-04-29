# ベースイメージ: TensorFlow 公式イメージの特定バージョンを使用 (CPU版)
FROM tensorflow/tensorflow:2.19.0

# タイムゾーン設定と追加で必要なOSパッケージ (tzdata) のインストール
# ベースイメージによっては他のパッケージが不要な場合もある
RUN apt-get update -qq && \
    apt-get install -y --no-install-recommends tzdata locales && \
    # タイムゾーンを Asia/Tokyo に設定
    ln -sf /usr/share/zoneinfo/Asia/Tokyo /etc/localtime && \
    echo "Asia/Tokyo" > /etc/timezone && \
    # locale 設定 (日本語が必要な場合)
    echo "ja_JP.UTF-8 UTF-8" >> /etc/locale.gen && \
    locale-gen ja_JP.UTF-8 && \
    # キャッシュ削除
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# 環境変数: タイムゾーンとロケール、Pythonログ設定
ENV TZ=Asia/Tokyo \
    LANG=ja_JP.UTF-8 \
    LANGUAGE=ja_JP:ja \
    LC_ALL=ja_JP.UTF-8 \
    PYTHONUNBUFFERED=1

# 作業ディレクトリの定義と設定
ARG APP_DIR=/app
WORKDIR ${APP_DIR}

# 非rootユーザー 'tfuser' (UID 1001) を作成し、作業ディレクトリの所有者にする
# (TensorFlow公式イメージには既に 'tf' ユーザーが存在する場合があるため、
#  別の名前 'appuser' や 'tfuser' を使うか、既存ユーザーを使うか検討)
# ここでは新規に 'appuser' を作成する例を示す
RUN useradd -m -u 1001 appuser && \
    mkdir -p ${APP_DIR} && \
    chown -R appuser:appuser ${APP_DIR}

# 追加の Python 依存関係ファイルをコピー
COPY --chown=appuser:appuser requirements.txt ./

# 追加の依存関係を appuser としてインストール
# TensorFlow と NumPy はベースイメージに含まれるため、ここでは不要
USER appuser
RUN pip install --no-cache-dir -r requirements.txt

# アプリケーションコードはこの Dockerfile ではコピーしない
# (開発時は compose.yml の volumes でマウントされるため)
# 本番用イメージなど、コードをイメージに含める場合は以下を追加:
# COPY --chown=appuser:appuser logging.conf .
# COPY --chown=appuser:appuser sample.py .

# デフォルトのコンテナ起動コマンド (開発時は compose.yml の command で上書きされる想定)
# CMD ["python", "sample.py"] 