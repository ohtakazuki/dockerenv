from flask import Flask
import socket

app = Flask(__name__)

@app.route('/')
def hello():
    # コンテナ自身のホスト名を取得
    hostname = socket.gethostname()
    # HTMLでメッセージとホスト名を表示
    return f"<h1>Hello from Multi-Stage Build!</h1><p>Served by container: {hostname}</p>"

if __name__ == '__main__':
    # コンテナ内のすべてのネットワークインターフェースからアクセス可能にし、ポート5000で待機
    app.run(host='0.0.0.0', port=5000)