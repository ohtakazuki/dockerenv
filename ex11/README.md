# Docker入門

## docker-composeコマンド

1. プロジェクトの開始<br>
   docker-compose up -d

2. コンテナへの接続<br>
   docker-compose exec python /bin/bash

3. Pythonプログラムの実行<br>
   python sample1.py

4. コンテナから抜ける<br>
   exit

5. プロジェクトの終了<br>
   docker-compose down --rmi all
