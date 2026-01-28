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
from paper_trader import execute_trade, monitor_positions, get_open_positions

# --- 設定 ---
TIMEFRAME = '4h'
TARGET_ACTIVE_COINS = 10   # 目標：每一輪都要確保有 10 個有效幣種被分析
SCAN_POOL_SIZE = 50        # 候選池：一次抓 50 個，預防有幣被打槍，才有足夠的候補補上
CONFIDENCE_THRESHOLD = 40.0
MODELS_DIR = 'models/'
DATA_DIR = 'data/'

# 排除清單
EXCLUDE_SYMBOLS = ['USDT/USDT', 'USDC/USDT', 'FDUSD/USDT', 'TUSD/USDT', 'DAI/USDT', 'WBTC/USDT']

def get_market_candidates(limit=50):
    """
    從 Binance 抓取大量候選名單 (例如前 50 名)
    這樣就算中間有幣被過濾掉，後面還有候補可以補上
    """
    print(f"[SCANNER] 正在掃描市場前 {limit} 大熱門幣種...")
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
        # 排序並取前 N 名
        df = df.sort_values(by='volume', ascending=False).head(limit)
        return df['symbol'].tolist()
    except Exception as e:
        print(f"[ERROR] 掃描失敗: {e}")
        return []

def analyze_potential(symbol):
    """
    篩選 1: 波動率檢查
    """
    try:
        exchange = ccxt.binance()
        ohlcv = exchange.fetch_ohlcv(symbol, TIMEFRAME, limit=50)
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['volatility'] = (df['high'] - df['low']) / df['close']
        avg_volatility = df['volatility'].mean() * 100
        
        if avg_volatility < 0.5:
            print(f"[FILTER] {symbol} 波動率過低 ({avg_volatility:.2f}%)，跳過 (尋找下一個)。")
            return False
        return True
    except:
        return False

def ensure_model_exists(symbol):
    """
    篩選 2: 數據長度與模型訓練檢查
    如果訓練失敗 (例如數據太短)，回傳 False，讓主迴圈去找下一個幣
    """
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
            # 這裡就是關鍵：如果訓練失敗 (例如 FOGOUSDT 資料太少)，回傳 False
            print(f"[SKIP] {symbol} 訓練失敗或數據不足，跳過 (尋找下一個)。")
            return False
    return True

def analyze_market(symbol):
    """
    執行預測與下單 (包含策略與槓桿資訊)
    """
    clean_symbol = symbol.replace('/', '')
    model_path = os.path.join(MODELS_DIR, f"{clean_symbol}_model.keras")
    scaler_path = os.path.join(MODELS_DIR, f"{clean_symbol}_scaler.pkl")
    
    try:
        scaler = joblib.load(scaler_path)
        model = load_model(model_path)
        
        df = get_latest_data(symbol, TIMEFRAME)
        if df.empty: return
        
        # 取得預測資料與策略資訊
        X_input, market_info = process_data(df, scaler)
        
        prediction = model.predict(X_input, verbose=0)
        prob_buy  = prediction[0][1] * 100
        prob_sell = prediction[0][2] * 100
        action = np.argmax(prediction)
        
        # ✅ 從 market_info 讀取策略參數
        strat_type = market_info['type']     # "LONG_TERM" or "SHORT_TERM"
        leverage = market_info['leverage']   # e.g., 5
        tp_mult = market_info['tp_mult']
        sl_mult = market_info['sl_mult']
        current_price = market_info['close']
        atr = market_info['ATR']
        
        print(f"\n[ANALYSIS] 標的: {symbol}")
        print(f"   現價: {current_price:.4f} | ATR: {atr:.4f} | ADX: {market_info['adx']:.1f}")
        print(f"   策略: {strat_type} | 建議槓桿: {leverage}x")
        print(f"   信心: Buy({prob_buy:.1f}%) | Sell({prob_sell:.1f}%)")
        
        trade_action = None
        
        # 根據策略計算 TP/SL
        if action == 1 and prob_buy > CONFIDENCE_THRESHOLD:
            trade_action = "BUY"
            tp = current_price + (atr * tp_mult)
            sl = current_price - (atr * sl_mult)
            
        elif action == 2 and prob_sell > CONFIDENCE_THRESHOLD:
            trade_action = "SELL"
            tp = current_price - (atr * tp_mult)
            sl = current_price + (atr * sl_mult)
            
        if trade_action:
            print(f"   [SIGNAL] ★ 發現 {trade_action} 機會！準備下單...")
            
            # ✅ [關鍵修改] 呼叫下單函式，傳入策略與槓桿
            execute_trade(
                symbol=symbol, 
                action=trade_action, 
                price=current_price, 
                tp=tp, 
                sl=sl, 
                strategy=strat_type, # 傳入策略類型
                leverage=leverage    # 傳入槓桿倍數
            )
        else:
            print(f"   [DECISION] 觀望 (信心不足)")

    except Exception as e:
        print(f"[ERROR] 分析 {symbol} 時發生錯誤: {e}")

def fund_manager_cycle():
    print(f"\n[SYSTEM] {datetime.now()} - 開始新一輪資產配置...")
    
    # 1. 監控現有持倉 (平倉獲利)
    monitor_positions()
    
    # 2. 獲取大量候選名單 (一次抓 50 個，確保夠用)
    candidates = get_market_candidates(limit=SCAN_POOL_SIZE)
    
    # 3. 取得目前手上的持倉 (為了避免重複下單)
    open_positions = get_open_positions()
    
    print(f"[SYSTEM] 準備挑選 {TARGET_ACTIVE_COINS} 個有效標的進行分析...")
    
    # 計數器：紀錄我們已經成功分析了幾個幣
    processed_count = 0
    
    for symbol in candidates:
        # 如果已經達成目標數量 (例如 10 個)，就提早下班
        if processed_count >= TARGET_ACTIVE_COINS:
            print(f"[SYSTEM] 已完成 {TARGET_ACTIVE_COINS} 個標的的分析，結束本輪掃描。")
            break
            
        # --- 檢查 1: 是否已持倉 ---
        if symbol in open_positions:
            print(f"[SKIP] 已持有 {symbol} 部位，跳過開單分析 (佔用 1 個名額)。")
            # 雖然跳過分析，但也算是一個「被管理的標的」，所以計數器 +1
            processed_count += 1 
            continue
            
        print("-" * 50)
        print(f"[CHECK] 候選人 {processed_count + 1}/{TARGET_ACTIVE_COINS}: {symbol}")
        
        # --- 檢查 2: 波動率 (太低就找下一個) ---
        if not analyze_potential(symbol):
            continue # 不加 processed_count，直接進下一迴圈找替補
            
        # --- 檢查 3: 模型與數據長度 (訓練失敗就找下一個) ---
        if not ensure_model_exists(symbol):
            continue # 不加 processed_count，直接進下一迴圈找替補
            
        # --- 通過所有考驗，進行分析 ---
        analyze_market(symbol)
        
        # 成功分析完一個，計數器 +1
        processed_count += 1

    print("-" * 50)
    print(f"[SYSTEM] 本輪結束 (共處理 {processed_count} 個有效標的)，等待下一次喚醒。")

if __name__ == "__main__":
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(DATA_DIR, exist_ok=True)
    
    fund_manager_cycle()
    
    schedule.every(1).hours.do(fund_manager_cycle)
    
    while True:
        schedule.run_pending()
        time.sleep(1)