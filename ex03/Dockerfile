FROM ubuntu:24.04

# 環境変数: noninteractiveモードを指定し、tzdataインストール時の対話プロンプトを抑制
ENV DEBIAN_FRONTEND=noninteractive

# Nginxとタイムゾーンデータをインストール
# RUN命令を&&で繋げてレイヤー数を減らし、最後にaptキャッシュを削除してイメージサイズを削減
RUN apt-get update -qq \
 && apt-get install -y --no-install-recommends nginx tzdata \
 && rm -rf /var/lib/apt/lists/*

# 環境変数: タイムゾーンをアジア/東京に設定
ENV TZ="Asia/Tokyo"

# ポートの指定
EXPOSE 80

# ファイルのコピー
COPY src/index.html /var/www/html/

# Nginxの起動
CMD ["nginx", "-g", "daemon off;"]