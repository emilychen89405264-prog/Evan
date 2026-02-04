import ccxt
import pandas as pd
import pandas_ta as ta
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
from paper_trader import execute_trade as paper_execute, monitor_positions, get_open_positions
from bybit_executor import execute_real_trade, get_account_balance, get_ticker_price

# --- 設定 ---
TIMEFRAME = '4h'
TARGET_ACTIVE_COINS = 10  
SCAN_POOL_SIZE = 100      
CONFIDENCE_THRESHOLD = 50.0 
MODELS_DIR = 'models/'
DATA_DIR = 'data/'

# 排除清單
EXCLUDE_SYMBOLS = ['USDT/USDT', 'USDC/USDT', 'FDUSD/USDT', 'TUSD/USDT', 'DAI/USDT', 'WBTC/USDT']

def get_btc_trend():
    """
    判斷比特幣大盤趨勢
    """
    try:
        exchange = ccxt.binance()
        ohlcv = exchange.fetch_ohlcv('BTC/USDT', '4h', limit=210)
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['EMA_200'] = df.ta.ema(length=200)
        
        current_price = df['close'].iloc[-1]
        ema_200 = df['EMA_200'].iloc[-1]
        
        if pd.isna(ema_200): return "NEUTRAL"

        if current_price > ema_200: return "BULL"
        else: return "BEAR"
    except Exception as e:
        print(f"[警告] 無法獲取 BTC 趨勢: {e}")
        return "NEUTRAL"

def get_market_candidates(limit=100):
    print(f"[掃描] 正在掃描 Bybit 市場前 {limit} 大熱門幣種...")
    try:
        exchange = ccxt.bybit({'options': {'defaultType': 'linear'}})
        tickers = exchange.fetch_tickers()
        data = []
        
        for symbol, ticker in tickers.items():
            # Bybit 的合約符號通常是 BTC/USDT:USDT，我們過濾 USDT 結尾的
            if '/USDT' in symbol and symbol not in EXCLUDE_SYMBOLS:
                # 確保有交易量資訊
                quote_vol = ticker.get('quoteVolume') or 0
                data.append({
                    'symbol': symbol,
                    'volume': quote_vol
                })
        
        df = pd.DataFrame(data)
        # 按成交量排序
        df = df.sort_values(by='volume', ascending=False).head(limit)
        return df['symbol'].tolist()
    except Exception as e:
        print(f"[錯誤] 掃描失敗: {e}")
        return []

def analyze_potential(symbol):
    try:
        # ✅ 改用 Bybit 抓 K 線
        exchange = ccxt.bybit({'options': {'defaultType': 'linear'}})
        ohlcv = exchange.fetch_ohlcv(symbol, TIMEFRAME, limit=50)
        
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['volatility'] = (df['high'] - df['low']) / df['close']
        avg_volatility = df['volatility'].mean() * 100
        
        if avg_volatility < 0.5:
            print(f"[過濾] {symbol} 波動率過低 ({avg_volatility:.2f}%)，跳過。")
            return False
        return True
    except:
        return False

def ensure_model_exists(symbol):
    clean_symbol = symbol.split(':')[0].replace('/', '')
    model_path = os.path.join(MODELS_DIR, f"{clean_symbol}_model.keras")
    scaler_path = os.path.join(MODELS_DIR, f"{clean_symbol}_scaler.pkl")
    
    if not os.path.exists(model_path) or not os.path.exists(scaler_path):
        print(f"[經理人] 尚未擁有 {symbol} 的模型，啟動自動訓練流程...")
        success = run_pipeline_for_coin(symbol)
        if success:
            print(f"[成功] {symbol} 模型訓練完成。")
            return True
        else:
            print(f"[跳過] {symbol} 訓練失敗或數據不足。")
            return False
    return True

def analyze_market(symbol, btc_trend):
    """
    執行預測與下單 (含即時價格校正)
    """
    clean_symbol = symbol.split(':')[0].replace('/', '')
    
    model_path = os.path.join(MODELS_DIR, f"{clean_symbol}_model.keras")
    scaler_path = os.path.join(MODELS_DIR, f"{clean_symbol}_scaler.pkl")
    
    try:
        scaler = joblib.load(scaler_path)
        model = load_model(model_path)
        
        df = get_latest_data(symbol, TIMEFRAME)
        if df.empty: return False
        
        X_input, market_info = process_data(df, scaler)
        
        prediction = model.predict(X_input, verbose=0)
        prob_buy  = prediction[0][1] * 100
        prob_sell = prediction[0][2] * 100
        action = np.argmax(prediction)
        
        # 讀取資訊
        strat_type = market_info['type']
        leverage = market_info['leverage']
        tp_mult = market_info['tp_mult']
        sl_mult = market_info['sl_mult']
        current_price = market_info['close'] # 這是 CSV 裡的價格 (可能是舊的)
        atr = market_info['ATR']
        
        print(f"\n[分析] 標的: {symbol}")
        print(f"   現價: {current_price:.4f} | ATR: {atr:.4f} | 策略: {strat_type} ({leverage}x)")
        print(f"   信心: Buy({prob_buy:.1f}%) | Sell({prob_sell:.1f}%)")
        
        decision_msg = ""
        trade_action = None
        
        # 預設的 TP/SL (基於 CSV 價格，僅供參考或模擬用)
        tp = 0
        sl = 0

        # --- 訊號判斷 ---
        if action == 1: # AI 建議: BUY
            if prob_buy > CONFIDENCE_THRESHOLD:
                if btc_trend == "BEAR":
                    decision_msg = f"[過濾] 訊號 Buy 但 BTC 處於熊市 (EMA200之下)，取消"
                else:
                    trade_action = "BUY"
                    tp = current_price + (atr * tp_mult)
                    sl = current_price - (atr * sl_mult)
            else:
                decision_msg = f"[觀望] 買入訊號信心不足 ({prob_buy:.1f}% < {CONFIDENCE_THRESHOLD}%)"

        elif action == 2: # AI 建議: SELL
            if prob_sell > CONFIDENCE_THRESHOLD:
                if btc_trend == "BULL":
                    decision_msg = f"[過濾] 訊號 Sell 但 BTC 處於牛市 (EMA200之上)，取消"
                else:
                    trade_action = "SELL"
                    tp = current_price - (atr * tp_mult)
                    sl = current_price + (atr * sl_mult)
            else:
                decision_msg = f"[觀望] 賣出訊號信心不足 ({prob_sell:.1f}% < {CONFIDENCE_THRESHOLD}%)"
                
        else: # AI 建議: HOLD
            decision_msg = f"[觀望] AI 判斷目前應持倉觀望 (Hold)"

        # --- 執行或顯示結果 ---
        if trade_action:
            print(f"   [訊號] 發現 {trade_action} 機會！正在執行程序...")
            
            real_time_price = get_ticker_price(symbol)
            final_tp = tp
            final_sl = sl
            record_price = current_price # 預設紀錄價格
            
            if real_time_price:
                print(f"   [校正] CSV價格: {current_price:.4f} -> Bybit即時價: {real_time_price:.4f}")
                record_price = real_time_price # 更新紀錄價格為真實價格
                
                # 重新計算止盈止損
                if trade_action == "BUY":
                    final_tp = real_time_price + (atr * tp_mult)
                    final_sl = real_time_price - (atr * sl_mult)
                elif trade_action == "SELL":
                    final_tp = real_time_price - (atr * tp_mult)
                    final_sl = real_time_price + (atr * sl_mult)
            else:
                print(f"   [警告] 無法獲取即時價格，將沿用 CSV 價格嘗試下單...")
            # ==========================================

            # 1. 計算資金
            current_balance = get_account_balance()
            if current_balance <= 0: current_balance = 1000 
            
            position_size_usdt = current_balance * 0.1 
            real_order_success = False

            if position_size_usdt < 10: 
                print(f"[警告] 資金不足 (${position_size_usdt:.2f})，僅執行模擬交易。")
            else:
                # 2. 嘗試發送真實訂單 (使用校正後的 final_tp/final_sl)
                real_order_success = execute_real_trade(
                    symbol=symbol,
                    action=trade_action,
                    quantity_usdt=position_size_usdt,
                    leverage=leverage,
                    tp_price=final_tp, 
                    sl_price=final_sl  
                )

            # 3. 邏輯分支處理
            if real_order_success:
                print(f"   [成功] 實盤下單成功！同步記錄至 CSV。")
            else:
                print(f"   [降級] 實盤下單失敗/跳過 (可能流動性不足)，轉為純模擬交易 (Paper Trade)。")

            # 4. 記錄交易 (Paper Execute)
            # 不管實盤有沒有成功，都記錄下來
            paper_execute(
                symbol=symbol, 
                action=trade_action, 
                price=record_price, 
                tp=final_tp, 
                sl=final_sl, 
                strategy=strat_type, 
                leverage=leverage
            )
            
            # 5. 回傳 True (計數器 +1)
            return True

        else:
            print(f"   [決策] {decision_msg}")
            return False

    except Exception as e:
        print(f"[錯誤] 分析 {symbol} 時發生錯誤: {e}")
        return False

def fund_manager_cycle():
    print(f"\n[系統] {datetime.now()} - 開始新一輪資產配置...")
    
    monitor_positions()
    
    btc_trend = get_btc_trend()
    print(f"[市場] BTC 大盤趨勢: {btc_trend} (作為多空濾網)")
    
    # 擴大候選池到 100，避免因為過濾太嚴格而找不到 10 個
    candidates = get_market_candidates(limit=SCAN_POOL_SIZE)
    open_positions = get_open_positions()
    
    print(f"[系統] 正在搜尋 {TARGET_ACTIVE_COINS} 個可執行的交易訊號...")
    
    signals_found = 0 # 計數器：只計算「成功下單」的次數
    
    for symbol in candidates:
        # 1. 符號清洗 (Bybit 格式 vs 乾淨格式)
        clean_symbol = symbol.split(':')[0]
        
        # 2. 取得目前持倉清單 (從 paper_trader 或交易所取得)
        # 注意：這裡要確保 get_open_positions 回傳的是乾淨的符號清單
        current_positions = [p['symbol'].split(':')[0] for p in get_open_positions()]
        
        # 🔥 修正 1: 防止重複下單 (最重要的修正)
        if clean_symbol in current_positions:
            # 這裡可以選擇不印出來，或是用 debug level，避免洗版
            # print(f"[跳過] {clean_symbol} 已在持倉中，避免重複下單。")
            continue 
            
        # 3. 檢查波動率... (原本的代碼)
        if not analyze_potential(symbol):
            continue 
            
        # ... (中間省略) ...

        # 4. 在下單前檢查「最大持倉數量」
        # 假設我們最多只能持倉 10 檔，避免資金爆掉
        if len(current_positions) >= 10:
             print(f"[警告] 倉位已滿 (10/10)，停止開新倉。")
             break # 直接跳出迴圈，不再找新幣

        # 5. 進行分析並嘗試下單
        is_traded = analyze_market(symbol, btc_trend)
        
        if is_traded:
            signals_found += 1
            print(f"[進度] 已發現訊號: {signals_found}/{TARGET_ACTIVE_COINS}")
        else:
            # 雖然分析了，但沒下單，所以不計入 signals_found，繼續找下一個
            pass

    if signals_found < TARGET_ACTIVE_COINS:
        print(f"[警告] 候選池已耗盡。在 {SCAN_POOL_SIZE} 個幣種中僅找到 {signals_found} 個訊號。")

    print("-" * 50)
    print(f"[系統] 本輪結束，等待下一次喚醒。")

if __name__ == "__main__":
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(DATA_DIR, exist_ok=True)
    
    fund_manager_cycle()
    
    schedule.every(1).hours.do(fund_manager_cycle)
    
    while True:
        schedule.run_pending()
        time.sleep(1)