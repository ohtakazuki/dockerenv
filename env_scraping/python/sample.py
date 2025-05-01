import requests
from bs4 import BeautifulSoup
import datetime
import schedule
import time
import os.path
import logging
import logging.config
from logging import getLogger

# --- Logger の設定 ---
# logging.conf ファイルを読み込む (WORKDIR からの相対パス)
try:
    logging.config.fileConfig("logging.conf")
    logger = getLogger(__name__) # root ロガーではなく、モジュール名を指定
except Exception as e:
    print(f"Error loading logging configuration: {e}")
    # フォールバックとして基本的なロギングを設定 (任意)
    logging.basicConfig(level=logging.INFO, format='[%(asctime)s]%(levelname)-8s%(name)s(%(lineno)d): %(message)s')
    logger = getLogger(__name__)
# --------------------

# --- 定数 ---
SEARCH_WORD = 'japan'
BASE_URL = 'https://news.google.com/search'
HEADERS = { # ユーザーエージェントを設定 (一部サイトでブロック回避に役立つ場合がある)
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
}
OUTPUT_DIR = "./out" # 出力先ディレクトリ (WORKDIR からの相対パス)
INTERVAL_MINUTES = 2 # 実行間隔 (分)
# --------------------

# --- 出力ディレクトリ作成 ---
if not os.path.isdir(OUTPUT_DIR):
    try:
        os.makedirs(OUTPUT_DIR)
        logger.info(f"Created output directory: {OUTPUT_DIR}")
    except OSError as e:
        logger.error(f"Failed to create output directory {OUTPUT_DIR}: {e}")
        exit(1) # ディレクトリ作成に失敗したら終了
# --------------------

# --- スクレイピング関数 ---
def scrape_google_news():
    """Google News から指定された検索ワードのタイトルを取得しファイルに保存する"""
    target_url = f'{BASE_URL}?q={SEARCH_WORD}&hl=ja&gl=JP&ceid=JP:ja'
    logger.info(f"Attempting to scrape: {target_url}")

    try:
        # Webページを取得 (タイムアウトを設定)
        response = requests.get(target_url, headers=HEADERS, timeout=10)
        response.raise_for_status() # ステータスコード 2xx 以外は例外を発生させる
        html = response.text
        logger.debug(f"Successfully fetched HTML content (length: {len(html)})")

        # Webページを解析
        soup = BeautifulSoup(html, 'html.parser') # または 'lxml' (要インストール)

        # ニュースタイトルを取得 (より具体的なセレクタを使用)
        # セレクタはサイト構造の変更により調整が必要になる場合がある
        elems = soup.select('article div div a') 
        logger.info(f"Found {len(elems)} news articles.")

        if not elems:
            logger.warning("No news articles found with the specified selector.")
            return

        # 現在時刻でファイル名を生成
        timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        filename = os.path.join(OUTPUT_DIR, f"{SEARCH_WORD}_{timestamp}.txt")

        # ファイルにタイトルを書き込み
        written_count = 0
        with open(filename, mode='w', encoding='utf-8') as f: # エンコーディングを指定
            for e in elems:
                title = e.get_text(strip=True) # strip=True で前後の空白を除去
                if title: # 空のタイトルを除外
                    f.write(title + '\n')
                    written_count += 1
        
        if written_count > 0:
            logger.info(f"Successfully saved {written_count} titles to {filename}")
        else:
             logger.warning(f"Found elements but failed to extract/write titles to {filename}")

    except requests.exceptions.RequestException as e:
        logger.error(f"HTTP request failed: {e}")
    except Exception as e:
        logger.exception(f"An unexpected error occurred during scraping: {e}") # トレースバックも記録

# --- スケジューリング ---
logger.info(f"Scheduling job to run every {INTERVAL_MINUTES} minutes.")
# 初回実行
scrape_google_news() 
# 定期実行のスケジュール
schedule.every(INTERVAL_MINUTES).minutes.do(scrape_google_news)

# --- メインループ ---
logger.info("Starting main loop to run scheduled jobs...")
while True:
    schedule.run_pending()
    # CPU負荷を下げるため、待機時間を適切に設定 (1秒は短いかもしれない)
    # スケジュールの最小単位に合わせて調整
    time.sleep(max(1, schedule.idle_seconds() if schedule.get_jobs() else 60)) 