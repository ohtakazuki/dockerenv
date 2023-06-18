# Docker入門

## Dockerfileの記述例1：シンプルなPython環境

1. イメージのビルド<br>
   docker image build -t ex01/python:1.0 .

2. コンテナーの起動<br>
   docker container run -it --rm ex01/python:1.0 /bin/bash

3. コンテナーの終了<br>
   exit
