from flask import Flask
from flask import render_template
import mysql.connector

app = Flask(__name__)

@app.route('/')
def index():
  # mysqlからデータを取得
  conn = mysql.connector.connect(user='my', password='my', host='flask_db', database='my')
  cur = conn.cursor()
  cur.execute("select * from book")
  books = cur.fetchall()
  cur.close()
  conn.close()

  # テンプレートに受け渡し
  html = render_template('index.html', books=books)
  return html

if __name__ == '__main__':
  app.run(port=5000)