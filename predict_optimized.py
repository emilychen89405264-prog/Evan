import ccxt
import pandas as pd
import numpy as np
import pandas_ta as ta
import joblib  # 用來讀取訓練好的 Scaler
from tensorflow.keras.models import load_model

# --- 設定參數 ---
# 必須與訓練時完全一致，否則 AI 會錯亂
SYMBOL = 'ETH/USDT'
TIMEFRAME = '4h'
TIME_STEPS = 60 
FEATURE_COLS = ['RSI', 'Dist_EMA200', 'Dist_EMA50', 'Vol_Rel', 'ATR']

def get_latest_data():
    """從 Binance 抓取最新數據"""
    print(f"正在連線 Binance 抓取 {SYMBOL} 最新數據...")
    exchange = ccxt.binance()
    # 抓取 500 根確保指標計算穩定 (EMA200 需要較多數據)
    ohlcv = exchange.fetch_ohlcv(SYMBOL, TIMEFRAME, limit=500)
    
    df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
    return df

def process_data(df):
    """計算指標並整理格式"""
    # 1. 計算技術指標
    df['EMA_50'] = df.ta.ema(length=50)
    df['EMA_200'] = df.ta.ema(length=200)
    df['RSI'] = df.ta.rsi(length=14)
    df['ATR'] = df.ta.atr(length=14)

    # 2. 產生特徵 (Features)
    df['Dist_EMA200'] = (df['close'] - df['EMA_200']) / df['EMA_200']
    df['Dist_EMA50'] = (df['close'] - df['EMA_50']) / df['EMA_50']
    df['Vol_Rel'] = df['volume'] / df['volume'].rolling(window=20).mean()

    # 3. 清除空值
    df.dropna(inplace=True)
    
    # 4. 取最後 60 筆 (AI 的視窗大小)
    if len(df) < TIME_STEPS:
        raise ValueError("數據不足，無法進行預測")
        
    recent_df = df.tail(TIME_STEPS).copy()
    last_candle = recent_df.iloc[-1] # 最新的一根 K 線資訊
    
    # 5. 數據正規化 (關鍵步驟！)
    # 載入訓練時存好的 Scaler
    try:
        scaler = joblib.load('models/scaler.pkl')
    except FileNotFoundError:
        print("錯誤：找不到 'scaler.pkl'。請先執行 train_optimized.py")
        exit()
        
    X = recent_df[FEATURE_COLS].values
    
    # 使用訓練好的 scaler 進行轉換，而不是 fit_transform
    X_scaled = scaler.transform(X)
    
    # 6. 轉為 3D 格式 (1, 60, 5)
    X_final = np.array([X_scaled])
    
    return X_final, last_candle

if __name__ == "__main__":
    try:
        # 1. 載入最佳模型
        model_path = 'best_crypto_model.keras'
        print(f"正在載入 AI 模型: {model_path} ...")
        model = load_model(model_path)
        
        # 2. 獲取並處理數據
        df = get_latest_data()
        X_input, last_candle = process_data(df)
        
        # 3. 進行預測
        print("AI 正在分析市場結構...")
        prediction = model.predict(X_input)
        
        # 解析機率
        prob_hold = prediction[0][0] * 100
        prob_buy  = prediction[0][1] * 100
        prob_sell = prediction[0][2] * 100
        
        # --- 顯示精美的分析報告 ---
        print("\n" + "="*40)
        print(f"標的: {SYMBOL} ({TIMEFRAME})")
        print(f"當前價格: {last_candle['close']:.2f}")
        print(f"市場波動 (ATR): {last_candle['ATR']:.2f}")
        print(f"資料時間: {last_candle['datetime']}")
        print("-" * 40)
        print(f"AI 信心分佈:")
        print(f"觀望 (Hold): {prob_hold:.2f}%")
        print(f"做多 (Buy) : {prob_buy:.2f}%")
        print(f"做空 (Sell): {prob_sell:.2f}%")
        print("-" * 40)
        
        # 最終建議邏輯
        action = np.argmax(prediction)
        
        # 設定一個信心門檻，如果沒有超過 40% 的把握，就建議觀望
        CONFIDENCE_THRESHOLD = 40.0
        
        if action == 1 and prob_buy > CONFIDENCE_THRESHOLD:
            tp = last_candle['close'] + (last_candle['ATR'] * 2)
            sl = last_candle['close'] - (last_candle['ATR'] * 1)
            print("💡 決策建議：【強力買進 LONG】")
            print(f"建議止損 (SL): {sl:.2f}")
            print(f"目標止盈 (TP): {tp:.2f} (獲利因子 2.0)")
            
        elif action == 2 and prob_sell > CONFIDENCE_THRESHOLD:
            tp = last_candle['close'] - (last_candle['ATR'] * 2)
            sl = last_candle['close'] + (last_candle['ATR'] * 1)
            print("決策建議：【強力做空 SHORT】")
            print(f"建議止損 (SL): {sl:.2f}")
            print(f"目標止盈 (TP): {tp:.2f} (獲利因子 2.0)")
            
        else:
            print("決策建議：【觀望 HOLD】")
            print("原因: AI 對於方向沒有足夠的共識，或是機率分佈過於平均。")
            print("建議空手，等待更明確的訊號。")
            
        print("="*40)

    except Exception as e:
        print(f"發生未預期的錯誤: {e}")