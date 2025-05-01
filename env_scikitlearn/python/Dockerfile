# ベースイメージ: slim イメージを使用
FROM python:3.13-slim

# タイムゾーン設定と必要なパッケージのインストール
# tzdata: タイムゾーン情報 / locales: 地域設定 (任意、ログ等で必要なら)
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

# 環境変数: タイムゾーンとロケールを設定
ENV TZ=Asia/Tokyo \
    LANG=ja_JP.UTF-8 \
    LANGUAGE=ja_JP:ja \
    LC_ALL=ja_JP.UTF-8 \
    # Pythonのログ出力をバッファリングしない
    PYTHONUNBUFFERED=1 

# 作業ディレクトリの定義と設定
ARG APP_DIR=/app
WORKDIR ${APP_DIR}

# 非rootユーザー 'appuser' (UID 1001) を作成し、作業ディレクトリの所有者にする
RUN useradd -m -u 1001 appuser && \
    mkdir -p ${APP_DIR} && \
    chown -R appuser:appuser ${APP_DIR}

# 依存関係ファイルを先にコピー (Dockerキャッシュ活用)
COPY --chown=appuser:appuser requirements.txt ./

# 依存関係を appuser としてインストール
USER appuser
RUN pip install --no-cache-dir -r requirements.txt

# アプリケーションコードはこの Dockerfile ではコピーしない
# (開発時は compose.yml の volumes でマウントされるため)
# 本番用イメージなど、コードをイメージに含める場合は以下を追加:
# COPY --chown=appuser:appuser logging.conf .
# COPY --chown=appuser:appuser sample.py .
# COPY --chown=appuser:appuser src ./src # src ディレクトリを使う場合

# デフォルトのコンテナ起動コマンド (開発時は compose.yml の command で上書きされる想定)
# CMD ["python", "sample.py"] 