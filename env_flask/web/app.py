import os
import mysql.connector
from mysql.connector import Error # エラーハンドリング用
from flask import Flask, render_template, current_app, g # g をインポート

app = Flask(__name__)

# --- DB接続設定 (環境変数から取得) ---
# launch.json や compose.yml で設定された環境変数を読み込む
db_config = {
    'user': os.environ.get('MYSQL_USER', 'my'), # デフォルト値を設定可能
    'password': os.environ.get('MYSQL_PASSWORD', 'my'),
    'host': os.environ.get('MYSQL_HOST', 'db'), # compose のサービス名
    'database': os.environ.get('MYSQL_DATABASE', 'my'),
    'raise_on_warnings': True # 警告を例外として扱う (推奨)
}
# ------------------------------------

# --- DB接続/切断 (リクエストコンテキストを使用) ---
def get_db():
    """リクエスト内でDB接続を取得・再利用し、リクエスト終了時に閉じる"""
    if 'db' not in g:
        try:
            g.db = mysql.connector.connect(**db_config)
            app.logger.info("Database connection established.")
        except Error as err:
            app.logger.error(f"Error connecting to database: {err}")
            g.db = None # エラー時は None を設定
            # ここでエラーページを表示するなどの処理も可能
    return g.db

@app.teardown_appcontext
def teardown_db(exception=None):
    """リクエスト終了時にDB接続を閉じる"""
    db = g.pop('db', None)
    if db is not None:
        db.close()
        app.logger.info("Database connection closed.")
# ------------------------------------

@app.route('/')
def index():
    books = [] # エラー時用のデフォルト空リスト
    conn = get_db() # リクエストコンテキストから接続を取得

    if conn: # 接続が成功した場合のみクエリ実行
        try:
            with conn.cursor() as cur: # with を使うと自動でカーソルが閉じる
                cur.execute("SELECT id, title, insert_timestamp FROM book ORDER BY id")
                books = cur.fetchall()
                app.logger.info(f"Fetched {len(books)} books from database.")
        except Error as err:
            app.logger.error(f"Error executing query: {err}")
            # エラーメッセージをユーザーに表示するなどの処理も可能
            books = [] # エラー時は空リスト
    else:
        app.logger.error("Cannot execute query due to database connection failure.")
        # 接続失敗メッセージをユーザーに表示する処理も可能

    # テンプレートにデータを渡してレンダリング
    return render_template('index.html', books=books)

# app.run() はデバッグサーバー用。
# 本番環境では Gunicorn などの WSGI サーバーを使用する。
if __name__ == '__main__':
    # host='0.0.0.0' はコンテナ外からのアクセスに必要
    # debug=True は FLASK_ENV=development で自動的に有効化される
    app.run(host='0.0.0.0', port=5000) 