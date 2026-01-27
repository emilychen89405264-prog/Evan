import ccxt
import pandas as pd
import numpy as np
import os
import joblib
import schedule
import time
from datetime import datetime
from tensorflow.keras.models import load_model

# 匯入工具與交易模組
from trainer_interface import run_pipeline_for_coin
from predict_optimized import get_latest_data, process_data
# ✅ 新增：匯入模擬交易功能
from paper_trader import execute_trade, monitor_positions, get_open_positions

# --- 設定 ---
TIMEFRAME = '4h'
SCAN_TOP_N = 10
CONFIDENCE_THRESHOLD = 40.0
MODELS_DIR = 'models/'
DATA_DIR = 'data/'
MAX_POSITIONS = 5 # 最大持倉數

EXCLUDE_SYMBOLS = ['USDT/USDT', 'USDC/USDT', 'FDUSD/USDT', 'TUSD/USDT', 'DAI/USDT', 'WBTC/USDT']

def get_top_volume_coins(limit=10):
    print("[SCANNER] 正在掃描市場尋找熱門標的...")
    try:
        exchange = ccxt.binance()
        tickers = exchange.fetch_tickers()
        data = []
        for symbol, ticker in tickers.items():
            if '/USDT' in symbol and symbol not in EXCLUDE_SYMBOLS:
                data.append({
                    'symbol': symbol,
                    'volume': ticker['quoteVolume']
                })
        df = pd.DataFrame(data)
        df = df.sort_values(by='volume', ascending=False).head(limit)
        return df['symbol'].tolist()
    except Exception as e:
        print(f"[ERROR] 掃描失敗: {e}")
        return []

def analyze_potential(symbol):
    try:
        exchange = ccxt.binance()
        ohlcv = exchange.fetch_ohlcv(symbol, TIMEFRAME, limit=50)
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['volatility'] = (df['high'] - df['low']) / df['close']
        avg_volatility = df['volatility'].mean() * 100
        if avg_volatility < 0.5:
            print(f"[FILTER] {symbol} 波動率過低 ({avg_volatility:.2f}%)，跳過。")
            return False
        return True
    except:
        return False

def ensure_model_exists(symbol):
    clean_symbol = symbol.replace('/', '')
    model_path = os.path.join(MODELS_DIR, f"{clean_symbol}_model.keras")
    scaler_path = os.path.join(MODELS_DIR, f"{clean_symbol}_scaler.pkl")
    
    if not os.path.exists(model_path) or not os.path.exists(scaler_path):
        print(f"[MANAGER] 尚未擁有 {symbol} 的模型，啟動自動訓練流程...")
        success = run_pipeline_for_coin(symbol)
        if success:
            print(f"[SUCCESS] {symbol} 模型訓練完成。")
            return True
        else:
            print(f"[ERROR] {symbol} 訓練失敗，跳過。")
            return False
    return True

def analyze_market(symbol):
    """
    核心邏輯：載入模型 -> 預測 -> (如果有訊號) 下單
    """
    clean_symbol = symbol.replace('/', '')
    model_path = os.path.join(MODELS_DIR, f"{clean_symbol}_model.keras")
    scaler_path = os.path.join(MODELS_DIR, f"{clean_symbol}_scaler.pkl")
    
    try:
        scaler = joblib.load(scaler_path)
        model = load_model(model_path)
        
        df = get_latest_data(symbol, TIMEFRAME)
        if df.empty: return
        
        X_input, last_candle = process_data(df, scaler)
        prediction = model.predict(X_input, verbose=0)
        
        prob_buy  = prediction[0][1] * 100
        prob_sell = prediction[0][2] * 100
        action = np.argmax(prediction)
        
        print(f"\n[ANALYSIS] 標的: {symbol}")
        print(f"   現價: {last_candle['close']:.4f} | ATR: {last_candle['ATR']:.4f}")
        print(f"   信心: Buy({prob_buy:.1f}%) | Sell({prob_sell:.1f}%)")
        
        current_price = last_candle['close']
        atr = last_candle['ATR']
        
        # --- 決策執行區 ---
        trade_action = None
        
        if action == 1 and prob_buy > CONFIDENCE_THRESHOLD:
            trade_action = "BUY"
            tp = current_price + (atr * 2)
            sl = current_price - (atr * 1)
            
        elif action == 2 and prob_sell > CONFIDENCE_THRESHOLD:
            trade_action = "SELL"
            tp = current_price - (atr * 2)
            sl = current_price + (atr * 1)
            
        if trade_action:
            print(f"   [SIGNAL] ★ 發現 {trade_action} 機會！準備下單...")
            # ✅ 呼叫 Paper Trader 執行下單
            execute_trade(symbol, trade_action, current_price, tp, sl)
        else:
            print(f"[DECISION] 觀望 (信心不足)")

    except Exception as e:
        print(f"[ERROR] 分析 {symbol} 時發生錯誤: {e}")

def fund_manager_cycle():
    print(f"\n[SYSTEM] {datetime.now()} - 開始新一輪資產管理...")
    
    # 1. ✅ 先監控現有持倉 (看看有沒有賺錢平倉的)
    monitor_positions()
    
    # 2. 檢查持倉上限
    open_positions = get_open_positions() # 取得目前持有的幣種列表
    if len(open_positions) >= MAX_POSITIONS:
        print(f"[LIMIT] 目前持倉已滿 ({len(open_positions)}/{MAX_POSITIONS})，暫停掃描新機會。")
        print("-" * 50)
        return

    # 3. 掃描市場
    candidates = get_top_volume_coins(SCAN_TOP_N)
    
    for symbol in candidates:
        # ✅ 如果手上已經有這個幣的單，就跳過分析 (不重複下單)
        if symbol in open_positions:
            continue
            
        print("-" * 50)
        print(f"[CHECK] 正在檢查: {symbol}")
        
        if not analyze_potential(symbol):
            continue
            
        if ensure_model_exists(symbol):
            analyze_market(symbol)

    print("-" * 50)
    print("[SYSTEM] 本輪結束，等待下一次喚醒。")

if __name__ == "__main__":
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(DATA_DIR, exist_ok=True)
    
    fund_manager_cycle()
    
    schedule.every(4).hours.do(fund_manager_cycle)
    
    while True:
        schedule.run_pending()
        time.sleep(1)