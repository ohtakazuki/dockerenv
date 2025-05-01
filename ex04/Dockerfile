# --- ステージ 1: ビルドステージ (builder) ---
FROM python:3.13 AS builder

# 作業ディレクトリを設定
WORKDIR /app

# 依存関係定義ファイルをコピー
COPY requirements.txt .

# 依存関係をホイールファイルとしてビルド・保存
RUN pip wheel --no-cache-dir --wheel-dir /wheels -r requirements.txt

# アプリケーションコードをコピー
COPY app.py .

# --- ステージ 2: 最終ステージ (runtime) ---
FROM python:3.13-slim

# 非rootユーザーを作成・設定
RUN useradd -r --create-home appuser \
    && mkdir -p /app && chown -R appuser:appuser /app

# 作業ディレクトリを設定
WORKDIR /app

# ビルドステージから必要なアーティファクトのみをコピー
COPY --from=builder --chown=appuser:appuser /wheels /wheels
COPY --from=builder --chown=appuser:appuser /app/requirements.txt .
COPY --from=builder --chown=appuser:appuser /app/app.py .

# コピーしたホイールファイルから依存関係をインストール
RUN pip install --no-cache-dir --no-index --find-links=/wheels -r requirements.txt \
    && rm -rf /wheels

# 以降の命令は 'appuser' として実行
USER appuser

# アプリケーションがリッスンするポートを指定
EXPOSE 5000

# コンテナ起動時にアプリケーションを実行するコマンド
CMD ["python", "app.py"]