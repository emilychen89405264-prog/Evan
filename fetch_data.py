import ccxt
import pandas as pd
import time
from datetime import datetime
import os

def fetch_historical_data(symbol, timeframe, start_date):
    """
    從 Binance 抓取歷史數據並處理分頁，並自動儲存至對應資料夾
    """
    
    # --- 1. 處理符號兼容性 (新增這段) ---
    # 如果傳進來的是 Bybit 格式 (BTC/USDT:USDT)，我們要切掉後面，只留 BTC/USDT
    # 這樣 Binance 才看得懂
    trading_pair = symbol.split(':')[0] 
    
    # 1. 初始化交易所 (維持使用 Binance 抓數據，因為資料最全)
    exchange = ccxt.binance({
        'enableRateLimit': True, 
    })

    # 時間轉換
    since = exchange.parse8601(start_date)
    
    all_ohlcv = []
    print(f"開始抓取 {trading_pair} [{timeframe}] 數據 (來源: Binance)")
    print(f"起始時間: {start_date} (Timestamp: {since})")
    print("-" * 50)

    while True:
        try:
            # 2. 抓取數據 (注意：這裡要用處理過的 trading_pair)
            ohlcv = exchange.fetch_ohlcv(trading_pair, timeframe, since, limit=1000)
            
            if len(ohlcv) == 0:
                break 
            
            all_ohlcv += ohlcv
            
            # 更新時間游標
            last_timestamp = ohlcv[-1][0]
            since = last_timestamp + 1 
            
            # 顯示進度
            current_date_str = datetime.fromtimestamp(last_timestamp / 1000).strftime('%Y-%m-%d %H:%M')
            print(f"已抓取至: {current_date_str} | 累積筆數: {len(all_ohlcv)}")
            
            if len(ohlcv) < 1000:
                print("已抓取到最新數據，停止抓取。")
                break
                
            time.sleep(0.5) 
            
        except Exception as e:
            error_msg = str(e).lower()
            if "does not have market symbol" in error_msg or "bad symbol" in error_msg:
                print(f"[忽略] 幣安找不到交易對 {trading_pair}，直接跳過此幣種。")
                break  
            
            print(f"發生錯誤: {e}")
            print("休息 5 秒後重試...")
            time.sleep(5) 
            continue

    # 3. 轉換格式
    df = pd.DataFrame(all_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')

    # 4. 自動歸檔 (核心邏輯保留在這裡)
    # ✅ [修正] 這裡必須處理冒號，不然 Windows 會報錯
    clean_symbol = symbol.split(':')[0].replace('/', '') # BTC/USDT:USDT -> BTCUSDT
    
    # 設定路徑: data/BTCUSDT/
    dir_path = os.path.join('data', clean_symbol)
    os.makedirs(dir_path, exist_ok=True)
    
    # 設定檔名: data/BTCUSDT/4h_data.csv
    filename = os.path.join(dir_path, f"{timeframe}_data.csv")
    
    df.to_csv(filename, index=False)
    print(f"檔案已儲存至: {filename}")
    
    return df

def validate_data(df):
    """數據健檢"""
    # ... (這部分不用改，維持原樣) ...
    print("\n" + "="*20 + " 數據健檢報告 " + "="*20)
    missing_values = df.isnull().sum().sum()
    if missing_values == 0:
        print("缺失值檢查：通過 (無空值)")
    else:
        print(f"缺失值檢查：失敗 (發現 {missing_values} 個空值)")

    duplicates = df.duplicated(subset=['timestamp']).sum()
    if duplicates == 0:
        print("重複值檢查：通過 (無重複時間)")
    else:
        print(f"重複值檢查：失敗 (發現 {duplicates} 筆重複數據)")

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
    
    # 執行抓取 (函式內部已經會自動存檔了)
    df = fetch_historical_data(TARGET_COIN, TIMEFRAME, START_TIME)
    
    # 執行檢查
    validate_data(df)