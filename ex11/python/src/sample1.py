import psycopg
import os

# 環境変数からDATABASE_URLを取得
DATABASE_URL = os.environ.get('DATABASE_URL')

# DATABASE_URLが設定されていない場合はエラーメッセージを表示して終了
if DATABASE_URL is None:
    print("エラー: 環境変数 DATABASE_URL が設定されていません。")
    exit()

# dbに接続し、pg_userテーブルの内容を表示する
with psycopg.connect(DATABASE_URL) as conn:
  with conn.cursor() as cur:
    cur.execute('SELECT * FROM pg_user')
    print(cur.fetchall())
