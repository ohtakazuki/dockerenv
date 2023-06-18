# Docker入門

## Dockerfileの記述例3：CMDコマンドの使用

1. イメージのビルド<br>
   docker image build -t ex03/nginx:1.0 .

2. コンテナーの起動<br>
   docker run --name ex03_websv -d --rm -p 8080:80 ex03/nginx:1.0

3. ブラウザで以下のアドレスにアクセス<br>
   http://localhost:8080/

4. コンテナーの停止<br>
   docker container stop ex03_websv
