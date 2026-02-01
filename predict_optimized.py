import ccxt
import pandas as pd
import numpy as np
import pandas_ta as ta
import joblib
import os
from tensorflow.keras.models import load_model

# --- 全域配置 ---
FEATURE_COLS = ['RSI', 'Dist_EMA200', 'Dist_EMA50', 'Vol_Rel', 'ATR']
TIME_STEPS = 60 

def get_latest_data(symbol, timeframe='4h', limit=500):
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

def calculate_strategy(df, current_price):
    """
    ✅ [優化] 策略大腦：調整 ATR 倍數以優化盈虧比
    """
    adx_df = df.ta.adx(length=14)
    if adx_df is not None and not adx_df.empty:
        current_adx = adx_df['ADX_14'].iloc[-1]
    else:
        current_adx = 0
    
    current_atr = df['ATR'].iloc[-1]
    volatility_pct = (current_atr / current_price) * 100
    
    # --- 判斷長短線與盈虧比 ---
    if current_adx > 25:
        strategy_type = "LONG_TERM" 
        # [優化] 長線趨勢明確，拉大獲利空間，忍受一些回調
        tp_mult = 4.0  
        sl_mult = 2.0  
    else:
        strategy_type = "SHORT_TERM"
        # [優化] 盤整行情，止損要嚴格，止盈見好就收
        tp_mult = 2.5  
        sl_mult = 1.0  

    # --- 判斷槓桿 (風險平價) ---
    risk_constant = 5.0 
    if volatility_pct > 0:
        raw_leverage = risk_constant / volatility_pct
    else:
        raw_leverage = 1.0
        
    leverage = int(np.clip(raw_leverage, 1, 20))

    return {
        'type': strategy_type,
        'adx': current_adx,
        'volatility_pct': volatility_pct,
        'leverage': leverage,
        'tp_mult': tp_mult,
        'sl_mult': sl_mult
    }

def process_data(df, scaler):
    if df.empty:
        raise ValueError("傳入的數據為空")

    df['EMA_50'] = df.ta.ema(length=50)
    df['EMA_200'] = df.ta.ema(length=200)
    df['RSI'] = df.ta.rsi(length=14)
    df['ATR'] = df.ta.atr(length=14)

    df['Dist_EMA200'] = (df['close'] - df['EMA_200']) / df['EMA_200']
    df['Dist_EMA50'] = (df['close'] - df['EMA_50']) / df['EMA_50']
    df['Vol_Rel'] = df['volume'] / df['volume'].rolling(window=20).mean()

    df.dropna(inplace=True)
    
    if len(df) < TIME_STEPS:
        raise ValueError(f"數據不足 (需要 {TIME_STEPS} 筆，僅有 {len(df)} 筆)")
        
    recent_df = df.tail(TIME_STEPS).copy()
    last_candle = recent_df.iloc[-1] 
    
    X = recent_df[FEATURE_COLS].values
    X_scaled = scaler.transform(X)
    X_final = np.array([X_scaled])
    
    strategy_info = calculate_strategy(df, last_candle['close'])
    
    last_candle_dict = last_candle.to_dict()
    last_candle_dict.update(strategy_info)
    
    return X_final, last_candle_dict

# --- 獨立測試區 ---
if __name__ == "__main__":
    # 設定測試參數
    TEST_SYMBOL = 'ETH/USDT'
    SCALER_PATH = 'models/scaler.pkl' 
    MODEL_PATH = 'models/best_crypto_model.keras'

    try:
        print(f"[TEST] 啟動單機測試模式: {TEST_SYMBOL}")

        if not os.path.exists(SCALER_PATH) or not os.path.exists(MODEL_PATH):
            print(f"[ERROR] 找不到模型或 Scaler 檔案，請檢查路徑。")
            exit()

        print(f"[LOAD] 載入 Scaler: {SCALER_PATH}")
        scaler = joblib.load(SCALER_PATH)
        
        print(f"[LOAD] 載入模型: {MODEL_PATH}")
        model = load_model(MODEL_PATH)
        
        # 呼叫工具
        df = get_latest_data(TEST_SYMBOL)
        
        # 處理數據 (現在會回傳策略建議了)
        X_input, market_info = process_data(df, scaler)
        
        # 預測
        print("[AI] 正在分析市場結構...")
        prediction = model.predict(X_input, verbose=0)
        prob_buy = prediction[0][1] * 100
        
        print("-" * 40)
        print(f"[RESULT] {TEST_SYMBOL}")
        print(f"   買入信心: {prob_buy:.2f}%")
        print(f"   當前 ADX: {market_info['adx']:.2f} ({market_info['type']})")
        print(f"   波動率: {market_info['volatility_pct']:.2f}%")
        print(f"   建議槓桿: {market_info['leverage']}x")
        print("-" * 40)

    except Exception as e:
        print(f"[ERROR] 測試失敗: {e}")