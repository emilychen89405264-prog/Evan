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
    接收 symbol 參數，從 Binance 抓取數據
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

def calculate_strategy(df, current_price):
    """
    ✅ [新增功能] 策略大腦
    根據 ADX (趨勢強度) 判斷長短線
    根據 ATR% (波動率) 判斷槓桿倍數
    """
    # 1. 確保 ADX 已計算 (pandas_ta 的 ADX 預設長度為 14)
    # df.ta.adx() 會回傳一個 DataFrame 包含 ADX_14, DMP_14, DMN_14
    adx_df = df.ta.adx(length=14)
    
    # 取得最新的 ADX 值
    if adx_df is not None and not adx_df.empty:
        current_adx = adx_df['ADX_14'].iloc[-1]
    else:
        current_adx = 0 # 預設值，避免報錯
    
    # 2. 計算波動率百分比 (ATR / Price)
    current_atr = df['ATR'].iloc[-1]
    volatility_pct = (current_atr / current_price) * 100
    
    # --- A. 判斷長短線 (基於 ADX) ---
    # ADX > 25 代表趨勢明確 -> 適合長線波段
    # ADX < 25 代表盤整震盪 -> 適合短線快進快出
    if current_adx > 25:
        strategy_type = "LONG_TERM" 
        tp_mult = 3.0  # 長線：止盈寬一點 (3倍 ATR)
        sl_mult = 1.5  # 長線：止損寬一點 (1.5倍 ATR)
    else:
        strategy_type = "SHORT_TERM"
        tp_mult = 1.5  # 短線：止盈窄一點 (1.5倍 ATR)
        sl_mult = 1.0  # 短線：止損緊一點 (1倍 ATR)

    # --- B. 判斷槓桿 (基於風險平價 Risk Parity) ---
    # 公式：目標風險係數 / 波動率
    # 假設我們希望每筆交易的波動風險控制在一定範圍內
    # 波動越小(如 BTC)，槓桿開大；波動越大(如 MEME)，槓桿開小
    
    risk_constant = 5.0 # 風險常數 (可自行調整)
    
    if volatility_pct > 0:
        raw_leverage = risk_constant / volatility_pct
    else:
        raw_leverage = 1.0
        
    # 限制槓桿範圍 (最低 1x，最高 20x)
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
    """
    接收 scaler 參數，計算指標並正規化，同時附加策略建議
    """
    if df.empty:
        raise ValueError("傳入的數據為空")

    # 1. 計算技術指標 (邏輯必須與訓練時完全一致)
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
    X = recent_df[FEATURE_COLS].values
    X_scaled = scaler.transform(X)
    
    # 6. 轉為 3D 格式 (1, 60, 5)
    X_final = np.array([X_scaled])
    
    # ==========================================
    # ✅ [修改點] 呼叫策略計算模組
    # ==========================================
    strategy_info = calculate_strategy(df, last_candle['close'])
    
    # 將策略資訊合併到 last_candle 裡回傳
    # 我們把 Series 轉成 Dict，這樣可以放更多自定義欄位
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