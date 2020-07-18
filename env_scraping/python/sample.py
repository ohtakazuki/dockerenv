import requests
from bs4 import BeautifulSoup
import datetime
import schedule
import time
import os.path

# 検索ワード
searchword = 'japan'

# WebサイトのURLを指定
url = f'https://news.google.com/search?q={searchword}&hl=ja&gl=JP&ceid=JP:ja'

# フォルダが無ければ作る
if os.path.isdir("out") == False:
    os.mkdir("out")

# スクレイピングを行う関数
def getnews():
  # Requestsを利用してWebページを取得する
  r = requests.get(url)
  html = r.text

  # BeautifulSoupを利用してWebページを解析する
  soup = BeautifulSoup(html, 'html.parser')

  # soup.selectを利用して、ニュースのタイトルを取得する
  elems = soup.select('article h3 a')

  fn = f'./out/{datetime.datetime.now().strftime("%Y%m%d-%H%M%S")}.txt'

  # ニュースのタイトルを出力する
  with open(fn, mode='w') as f:
    for e in elems:
      f.write(e.getText() + '\n')

# 2分おきにスケジュールを定義
schedule.every(2).minutes.do(getnews)

# 実行を監視し、指定時間になったら関数を実行する
while True:
    schedule.run_pending()
    time.sleep(1)