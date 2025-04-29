# ベースイメージ: 特定バージョンの Python slim イメージを使用
FROM python:3.13-slim

# 引数で作業ディレクトリを定義 (変更しやすくするため)
ARG APP_DIR=/app
WORKDIR ${APP_DIR}

# 非rootユーザー 'appuser' (UID 1001) を作成し、作業ディレクトリの所有者にする
# UIDを固定するとホストとのファイルパーミッション問題が起きにくい場合がある
RUN useradd -m -u 1001 appuser && \
    # APP_DIR が存在しない場合に作成 (-p オプション)
    mkdir -p ${APP_DIR} && \
    chown -R appuser:appuser ${APP_DIR}

# 依存関係ファイルを先にコピー (Dockerキャッシュを活用するため)
# 所有者を appuser に指定
COPY --chown=appuser:appuser requirements.txt ./

# 依存関係を appuser としてインストール (--no-cache-dir でイメージサイズ削減)
USER appuser
RUN pip install --no-cache-dir -r requirements.txt

# Python の標準出力をバッファリングしない設定 (ログが見やすくなる)
ENV PYTHONUNBUFFERED=1

# アプリケーションコードはこの Dockerfile ではコピーしない
# (開発時は compose.yml の volumes でマウントされるため)
# 本番用イメージなど、コードをイメージに含める場合は以下の行を追加:
# COPY --chown=appuser:appuser src ./src

# デフォルトのコンテナ起動コマンド (開発時は compose.yml の command で上書きされる想定)
# CMD ["python", "src/sample1.py"]