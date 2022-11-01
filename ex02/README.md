# Docker入門

## Dockerfileの記述例2：RUNコマンドの使用

1. イメージのビルド<br>
   docker image build --build-arg wdir=/tmp/src -t ex02/python:1.0 .

2. コンテナの起動<br>
   docker container run -it --rm --mount type=bind,src=$(pwd)/src,dst=/tmp/src ex02/python:1.0 /bin/bash

3. Pythonプログラムの実行<br>
   python sample1.py

4. コンテナの終了<br>
   exit


