# ベースイメージ: slim イメージを使用
FROM python:3.13-slim

# タイムゾーン設定と必要な OS パッケージ (locales 等)
# sentencepiece のビルドに必要な cmake, pkg-config, build-essential を追加
RUN apt-get update -qq && \
    apt-get install -y --no-install-recommends tzdata locales curl \
    cmake pkg-config build-essential && \ 
    # タイムゾーン設定
    ln -sf /usr/share/zoneinfo/Asia/Tokyo /etc/localtime && \
    echo "Asia/Tokyo" > /etc/timezone && \
    # ロケール設定
    echo "ja_JP.UTF-8 UTF-8" >> /etc/locale.gen && \
    locale-gen ja_JP.UTF-8 && \
    # キャッシュ削除
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# 環境変数: 基本設定
ENV TZ=Asia/Tokyo \
    LANG=ja_JP.UTF-8 \
    LANGUAGE=ja_JP:ja \
    LC_ALL=ja_JP.UTF-8 \
    PYTHONUNBUFFERED=1

# 作業ディレクトリ設定 (後のステップで USER 変更後に HOME を使うため早めに定義)
ARG APP_DIR=/app
ARG USERNAME=appuser
ARG USER_UID=1000
ARG USER_GID=$USER_UID

# 非 root ユーザー 'appuser' を作成
RUN groupadd --gid $USER_GID $USERNAME && \
    useradd --uid $USER_UID --gid $USER_GID -m $USERNAME && \
    # (任意) sudo 権限が必要なら追加
    # apt-get update && apt-get install -y sudo && \
    # echo $USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME && \
    # chmod 0440 /etc/sudoers.d/$USERNAME && \
    # apt-get clean && rm -rf /var/lib/apt/lists/* && \
    # 作業ディレクトリ作成と権限設定
    mkdir -p ${APP_DIR} && \
    chown -R $USERNAME:$USERNAME ${APP_DIR}

# --- Hugging Face キャッシュディレクトリ設定 ---
# ユーザーのホームディレクトリ以下に作成し、権限を設定
USER $USERNAME
ENV HF_HOME=/home/$USERNAME/.cache/huggingface
ENV TRANSFORMERS_CACHE=$HF_HOME/transformers
RUN mkdir -p $HF_HOME && \
    chown -R $USERNAME:$USERNAME /home/$USERNAME/.cache

# 作業ディレクトリを /app に設定
WORKDIR ${APP_DIR}

# 依存関係ファイルを先にコピー (権限も設定)
COPY --chown=$USERNAME:$USERNAME requirements.txt ./

# --- Python パッケージインストール ---
# CPU 版 PyTorch を明示的にインストール (環境によっては自動検出でも可)
# RUN pip install torch --index-url https://download.pytorch.org/whl/cpu --no-cache-dir
# requirements.txt に torch を含める場合は上記は不要なことが多い

# requirements.txt に記載されたライブラリをインストール
RUN pip install --no-cache-dir -r requirements.txt

# --- GPU 利用時の補足 ---
# GPU を利用する場合:
# 1. ベースイメージを NVIDIA CUDA 対応イメージに変更する。
#    例: FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04
#    (利用したい PyTorch/TensorFlow/CUDA バージョンに合わせる)
# 2. Python や pip をインストールする必要がある場合がある。
# 3. requirements.txt で GPU 版 PyTorch を指定する。
#    例: torch==2.1.0+cu118 -f https://download.pytorch.org/whl/torch_stable.html
# 4. 量子化ライブラリ (bitsandbytes など) は GPU がほぼ必須。
#    pip install bitsandbytes (ビルドが必要な場合、依存関係も要確認)

# アプリケーションコードは開発時はマウントされるため、ここではコピーしない
# COPY --chown=$USERNAME:$USERNAME . ${APP_DIR}

# デフォルトコマンド (compose.yml で上書きされる想定)
# CMD ["python", "sample.py"]
