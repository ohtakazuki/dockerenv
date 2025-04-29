# --- ステージ 1: ビルドステージ (builder) ---
# ビルドに必要なツールをインストールするため、slimイメージをベースにする
FROM python:3.13-slim AS builder

# ビルドに必要なAPTパッケージをインストール
# (apt-get update と install を1つのRUN命令にまとめ、最後にクリーンアップ)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# 作業ディレクトリを設定
WORKDIR /app

# 依存関係ファイルのみ先にコピー (キャッシュ効率化のため)
COPY requirements.txt .

# requirements.txt に基づいてホイールファイルを作成し、/wheels に保存
RUN pip wheel --no-cache-dir -r requirements.txt -w /wheels

# --- ステージ 2: 最終ステージ (runtime) ---
FROM python:3.13-slim

# 非rootユーザーを作成・設定
RUN useradd -r --create-home appuser \
    && mkdir -p /app && chown -R appuser:appuser /app

# psycopg の実行時に必要な libpq5 のみをインストール
RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq5 \
    && rm -rf /var/lib/apt/lists/*

# 作業ディレクトリを設定
WORKDIR /app

# ビルドステージからホイールファイル群をコピー
COPY --from=builder --chown=appuser:appuser /wheels /wheels

# コピーしたホイールファイルを使って依存関係をインストール
RUN pip install --no-cache-dir --no-index --find-links=/wheels /wheels/*.whl \
    && rm -rf /wheels

# アプリケーションコードをコピー
COPY --chown=appuser:appuser src ./src

# 以降の命令は 'appuser' として実行
USER appuser

# Pythonの出力をバッファリングしないように設定 (ログが見やすくなる)
ENV PYTHONUNBUFFERED=1

# コンテナ起動時にデフォルトで実行されるコマンドを設定
# (compose.yml の command を削除した場合にこれが使われる)
# CMD ["python", "src/sample1.py"]