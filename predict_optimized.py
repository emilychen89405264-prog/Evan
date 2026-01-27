import ccxt
import pandas as pd
import numpy as np
import pandas_ta as ta
import joblib
import os
from tensorflow.keras.models import load_model

# --- 全域配置 ---
# 這些特徵欄位必須與訓練時完全一致，不能改
FEATURE_COLS = ['RSI', 'Dist_EMA200', 'Dist_EMA50', 'Vol_Rel', 'ATR']
TIME_STEPS = 60 

def get_latest_data(symbol, timeframe='4h', limit=500):
    """
    [修改點 1] 接收 symbol 參數
    不再寫死 'ETH/USDT'，而是讓 Fund Manager 告訴我要抓誰
    """
    print(f"[NET] 正在連線 Binance 抓取 {symbol} 最新數據...")
    try:
        exchange = ccxt.binance()
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
        
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
        return df
    except Exception as e:
        print(f"[ERROR] 抓取數據失敗: {e}")
        return pd.DataFrame()

def process_data(df, scaler):
    """
    [修改點 2] 接收 scaler 參數 (依賴注入)
    因為 BTC 的 scaler 和 ETH 的 scaler 不同，必須由外部傳入正確的那一個
    """
    if df.empty:
        raise ValueError("傳入的數據為空")

    # 1. 計算技術指標 (邏輯必須與訓練時 add_indicators.py 完全一致)
    df['EMA_50'] = df.ta.ema(length=50)
    df['EMA_200'] = df.ta.ema(length=200)
    df['RSI'] = df.ta.rsi(length=14)
    df['ATR'] = df.ta.atr(length=14)

    # 2. 產生特徵
    df['Dist_EMA200'] = (df['close'] - df['EMA_200']) / df['EMA_200']
    df['Dist_EMA50'] = (df['close'] - df['EMA_50']) / df['EMA_50']
    df['Vol_Rel'] = df['volume'] / df['volume'].rolling(window=20).mean()

    # 3. 清除空值
    df.dropna(inplace=True)
    
    # 4. 檢查數據長度
    if len(df) < TIME_STEPS:
        raise ValueError(f"數據不足 (需要 {TIME_STEPS} 筆，僅有 {len(df)} 筆)")
        
    # 取最後 N 筆 (AI 的視窗)
    recent_df = df.tail(TIME_STEPS).copy()
    last_candle = recent_df.iloc[-1] 
    
    # 5. 數據正規化
    # [修改點 3] 直接使用傳入的 scaler，不再自己 load
    X = recent_df[FEATURE_COLS].values
    X_scaled = scaler.transform(X)
    
    # 6. 轉為 3D 格式 (1, 60, 5)
    X_final = np.array([X_scaled])
    
    return X_final, last_candle

# --- 獨立測試區 ---
# 這段程式碼只有當你直接執行 python predict_optimized.py 時才會跑
# 如果是被 fund_manager import 呼叫，這段不會執行
if __name__ == "__main__":
    # 設定測試參數 (模擬 Fund Manager 的行為)
    TEST_SYMBOL = 'ETH/USDT'
    SCALER_PATH = 'models/scaler.pkl' 
    MODEL_PATH = 'models/best_crypto_model.keras'

    try:
        print(f"[TEST] 啟動單機測試模式: {TEST_SYMBOL}")

        # 1. 模擬載入資源
        if not os.path.exists(SCALER_PATH) or not os.path.exists(MODEL_PATH):
            print(f"[ERROR] 找不到模型或 Scaler 檔案，請檢查路徑。")
            exit()

        print(f"[LOAD] 載入 Scaler: {SCALER_PATH}")
        scaler = joblib.load(SCALER_PATH)
        
        print(f"[LOAD] 載入模型: {MODEL_PATH}")
        model = load_model(MODEL_PATH)
        
        # 2. 呼叫工具 (傳入參數)
        df = get_latest_data(TEST_SYMBOL)
        
        # 3. 處理數據 (傳入 scaler)
        X_input, last_candle = process_data(df, scaler)
        
        # 4. 預測
        print("[AI] 正在分析市場結構...")
        prediction = model.predict(X_input, verbose=0)
        
        # 解析結果
        prob_buy  = prediction[0][1] * 100
        
        print(f"[RESULT] {TEST_SYMBOL} 買入信心: {prob_buy:.2f}%")

    except Exception as e:
        print(f"[ERROR] 測試失敗: {e}")