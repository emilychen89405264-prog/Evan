import ccxt
import pandas as pd
import time
from datetime import datetime

def fetch_historical_data(symbol, timeframe, start_date):
    """
    從 Binance 抓取歷史數據並處理分頁
    :param symbol: 交易對，例如 'ETH/USDT'
    :param timeframe: 時間週期，例如 '4h'
    :param start_date: 開始時間字串，例如 '2022-01-01 00:00:00'
    :return: 包含數據的 Pandas DataFrame
    """
    
    # 1. 初始化交易所 (使用 Binance 公開 API)
    # enableRateLimit=True 會讓程式自動調節請求速度，避免被鎖 IP
    exchange = ccxt.binance({
        'enableRateLimit': True, 
    })

    # 將字串時間轉換為毫秒時間戳 (Unix Timestamp)
    since = exchange.parse8601(start_date)
    
    all_ohlcv = []
    print(f"開始抓取 {symbol} [{timeframe}] 數據")
    print(f"起始時間: {start_date} (Timestamp: {since})")
    print("-" * 50)

    while True:
        try:
            # 2. 抓取數據 (Binance 單次上限通常為 1000 筆)
            # print(f"正在抓取自 {datetime.fromtimestamp(since/1000)} 以後的數據...")
            ohlcv = exchange.fetch_ohlcv(symbol, timeframe, since, limit=1000)
            
            # 如果抓不到數據了，就跳出迴圈
            if len(ohlcv) == 0:
                break 
            
            all_ohlcv += ohlcv
            
            # 取得這一批數據最後一筆的時間，並 +1 毫秒作為下一次抓取的起點
            last_timestamp = ohlcv[-1][0]
            since = last_timestamp + 1 
            
            # 顯示當前進度
            current_date_str = datetime.fromtimestamp(last_timestamp / 1000).strftime('%Y-%m-%d %H:%M')
            print(f"已抓取至: {current_date_str} | 累積筆數: {len(all_ohlcv)}")
            
            # 如果抓到的數據少於 limit (代表已經是最新的一批了)，就結束
            if len(ohlcv) < 1000:
                print("已抓取到最新數據，停止抓取。")
                break
                
            # 3. 休息一下，雖然有 enableRateLimit，但手動 sleep 更保險
            time.sleep(0.5) 
            
        except Exception as e:
            print(f"發生錯誤: {e}")
            print("休息 5 秒後重試...")
            time.sleep(5) 
            continue

    # 4. 轉換為 DataFrame 表格格式
    df = pd.DataFrame(all_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    
    # 將時間戳轉換為人類可讀的時間格式
    df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
    
    return df

def validate_data(df):
    """
    對下載後的數據進行健檢
    """
    print("\n" + "="*20 + " 數據健檢報告 " + "="*20)
    
    # 1. 檢查缺失值
    missing_values = df.isnull().sum().sum()
    if missing_values == 0:
        print("缺失值檢查：通過 (無空值)")
    else:
        print(f"缺失值檢查：失敗 (發現 {missing_values} 個空值)")
        print(df.isnull().sum())

    # 2. 檢查重複值
    duplicates = df.duplicated(subset=['timestamp']).sum()
    if duplicates == 0:
        print("重複值檢查：通過 (無重複時間)")
    else:
        print(f"重複值檢查：失敗 (發現 {duplicates} 筆重複數據)")

    # 3. 檢查時間範圍
    start_time = df['datetime'].iloc[0]
    end_time = df['datetime'].iloc[-1]
    print(f"數據時間範圍：{start_time} 至 {end_time}")
    print(f"總資料筆數：{len(df)} 筆")
    print("="*54 + "\n")


if __name__ == "__main__":
    
    # 設定參數
    TARGET_COIN = 'ETH/USDT'
    TIMEFRAME = '4h'          
    START_TIME = '2021-01-01 00:00:00' 
    
   
    df = fetch_historical_data(TARGET_COIN, TIMEFRAME, START_TIME)
    
    
    validate_data(df)
    
    # 存檔
    filename = f"data/{TARGET_COIN.replace('/', '')}_{TIMEFRAME}_data.csv"
    df.to_csv(filename, index=False)
    print(f"檔案已儲存為: {filename}")                                                                                  