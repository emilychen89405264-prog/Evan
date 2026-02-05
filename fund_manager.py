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

# 改用 Binance 執行器
from binance_executor import execute_real_trade, get_account_balance, get_ticker_price

# --- 設定 ---
TIMEFRAME = '4h'
TARGET_ACTIVE_COINS = 10 
SCAN_POOL_SIZE = 100       
CONFIDENCE_THRESHOLD = 50.0 
MODELS_DIR = 'models/'
DATA_DIR = 'data/'

# 排除清單
EXCLUDE_SYMBOLS = ['USDT/USDT', 'USDC/USDT', 'FDUSD/USDT', 'TUSD/USDT', 'DAI/USDT', 'WBTC/USDT']

# ==========================================
# 🛠️ 助手函式：強制連線到真實主網 (Mainnet)
# ==========================================
def get_mainnet_binance(options=None):
    """
    建立一個 ccxt Binance 物件，並強制指定 URL 為真實世界網址。
    這是為了防止 ccxt 偷偷連回測試網 (Testnet)。
    """
    config = {
        'options': {'defaultType': 'future'}, # 預設為合約
        'urls': {
            'api': {
                'fapiPublic': 'https://fapi.binance.com/fapi/v1', 
                'fapiPrivate': 'https://fapi.binance.com/fapi/v1',
            }
        }
    }
    if options:
        config['options'].update(options)
    
    exchange = ccxt.binance(config)
    exchange.set_sandbox_mode(False) # 再次確認關閉沙盒
    return exchange

def get_btc_trend():
    """ 判斷比特幣大盤趨勢 """
    try:
        # 使用助手函式連線 (強制主網)
        # 這裡用 spot (現貨) 或是 future (合約) 看趨勢都可以，這裡我們用合約看
        exchange = get_mainnet_binance()
        
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
    print(f"[掃描] 正在從 Binance 合約主網掃描前 {limit} 大熱門幣種...")
    try:
        exchange = get_mainnet_binance() # 強制主網
        tickers = exchange.fetch_tickers()
        data = []
        for symbol, ticker in tickers.items():
            if '/USDT' in symbol and symbol not in EXCLUDE_SYMBOLS:
                quote_vol = ticker.get('quoteVolume') or 0
                clean_symbol = symbol.split(':')[0] 
                data.append({'symbol': clean_symbol, 'volume': quote_vol})
        df = pd.DataFrame(data)
        df = df.sort_values(by='volume', ascending=False).head(limit)
        return df['symbol'].tolist()
    except Exception as e:
        print(f"[錯誤] 掃描失敗: {e}")
        return []

def analyze_potential(symbol):
    try:
        exchange = get_mainnet_binance() # 強制主網
        ohlcv = exchange.fetch_ohlcv(symbol, TIMEFRAME, limit=50)
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['volatility'] = (df['high'] - df['low']) / df['close']
        avg_volatility = df['volatility'].mean() * 100
        if avg_volatility < 0.5: return False
        return True
    except: return False

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

# ==========================================
# 🔥 [核心演算法] 波動率目標定位 (Risk Parity)
# ==========================================
def calculate_dynamic_leverage(atr, current_price):
    """
    目標：讓每一筆交易的「價格波動風險」都維持在 8% 左右。
    公式：Leverage = 8 / ATR%
    """
    if current_price == 0: return 1
    
    # 1. 計算 ATR 百分比
    atr_pct = (atr / current_price) * 100
    
    # 2. 設定目標波動率 (Target Volatility)
    target_volatility = 8.0 
    
    # 3. 計算槓桿
    if atr_pct <= 0: return 1
    raw_leverage = target_volatility / atr_pct
    
    # 4. 安全截斷 (Min 1x, Max 20x)
    final_leverage = max(1, min(int(raw_leverage), 20))
    
    return final_leverage, atr_pct

def analyze_market(symbol, btc_trend):
    clean_symbol = symbol.split(':')[0].replace('/', '')
    model_path = os.path.join(MODELS_DIR, f"{clean_symbol}_model.keras")
    scaler_path = os.path.join(MODELS_DIR, f"{clean_symbol}_scaler.pkl")
    
    try:
        scaler = joblib.load(scaler_path)
        model = load_model(model_path)
        
        # 這裡其實也會用到 ccxt，請確保 get_latest_data 內部也是連線正常的
        # 但通常 get_latest_data 是抓歷史 K 線，影響較小，關鍵是下面的即時查價
        df = get_latest_data(symbol, TIMEFRAME) 
        if df.empty: return False
        
        X_input, market_info = process_data(df, scaler)
        prediction = model.predict(X_input, verbose=0)
        prob_buy  = prediction[0][1] * 100
        prob_sell = prediction[0][2] * 100
        action = np.argmax(prediction)
        
        strat_type = market_info['type']
        current_price = market_info['close'] 
        atr = market_info['ATR']
        
        # ✅ 計算動態槓桿
        dynamic_leverage, atr_pct = calculate_dynamic_leverage(atr, current_price)

        print(f"\n[分析] 標的: {symbol}")
        print(f"   CSV價: {current_price:.4f} | ATR: {atr_pct:.2f}% | 策略: {strat_type}")
        print(f"   信心: Buy({prob_buy:.1f}%) Sell({prob_sell:.1f}%) -> 建議槓桿: {dynamic_leverage}x")
        
        decision_msg = ""
        trade_action = None
        tp = 0; sl = 0

        # --- 訊號判斷 ---
        if action == 1: # BUY
            if prob_buy > CONFIDENCE_THRESHOLD:
                if btc_trend == "BEAR": decision_msg = f"[過濾] 訊號 Buy 但 BTC 熊市"
                else: trade_action = "BUY"
            else: decision_msg = f"[觀望] 買入信心不足"
        elif action == 2: # SELL
            if prob_sell > CONFIDENCE_THRESHOLD:
                if btc_trend == "BULL": decision_msg = f"[過濾] 訊號 Sell 但 BTC 牛市"
                else: trade_action = "SELL"
            else: decision_msg = f"[觀望] 賣出信心不足"
        else: decision_msg = f"[觀望] 持倉觀望"

        # --- 執行下單 ---
        if trade_action:
            print(f"   [訊號] 發現 {trade_action} 機會！正在執行程序...")
            
            # 🔥 [關鍵防護] 再次查價，確保是真實價格
            real_time_price = get_ticker_price(symbol)
            final_price = current_price
            
            if real_time_price:
                # 🛡️ 安全檢查：如果 CSV 價格跟即時價格差太多 (>5%)，代表數據有問題，拒絕下單
                price_diff = abs(real_time_price - current_price) / current_price
                if price_diff > 0.05:
                    print(f"   [危險] 價格異常！CSV價: {current_price} vs 真實價: {real_time_price} (差距 {price_diff*100:.1f}%)")
                    print(f"   [中止] 可能是測試網數據殘留，取消本次下單。")
                    return False

                print(f"   [校正] CSV價格: {current_price:.4f} -> Binance真實價: {real_time_price:.4f}")
                final_price = real_time_price
                # 用真實價格重算槓桿
                dynamic_leverage, _ = calculate_dynamic_leverage(atr, final_price)
            else:
                print(f"   [警告] 連線 Binance 失敗，沿用 CSV 舊價格 (風險高)")

            # 計算 TP/SL
            tp_mult = market_info['tp_mult']
            sl_mult = market_info['sl_mult']

            if trade_action == "BUY":
                tp = final_price + (atr * tp_mult)
                sl = final_price - (atr * sl_mult)
            else:
                tp = final_price - (atr * tp_mult)
                sl = final_price + (atr * sl_mult)

            current_balance = get_account_balance()
            if current_balance <= 0: current_balance = 50000 
            
            position_size_usdt = current_balance * 0.1 
            real_order_success = False

            if position_size_usdt < 10: 
                print(f"[警告] 資金不足")
            else:
                # 傳入 Risk Parity 算出來的槓桿
                real_order_success = execute_real_trade(
                    symbol=symbol,
                    action=trade_action,
                    quantity_usdt=position_size_usdt,
                    leverage=dynamic_leverage, 
                    tp_price=tp, 
                    sl_price=sl  
                )

            if real_order_success: print(f"   [成功] 實盤下單成功！")
            else: print(f"   [降級] 實盤下單失敗，轉模擬。")

            paper_execute(
                symbol=symbol, 
                action=trade_action, 
                price=final_price,
                tp=tp, sl=sl, 
                strategy=strat_type, 
                leverage=dynamic_leverage
            )
            return True
        else:
            print(f"   [決策] {decision_msg}")
            return False

    except Exception as e:
        print(f"[錯誤] 分析 {symbol} 時發生錯誤: {e}")
        return False

def fund_manager_cycle():
    print(f"\n[系統] {datetime.now()} - 開始新一輪資產配置 (Risk Parity Mainnet)...")
    monitor_positions()
    btc_trend = get_btc_trend()
    print(f"[市場] BTC 大盤趨勢: {btc_trend}")
    candidates = get_market_candidates(limit=SCAN_POOL_SIZE)
    current_positions = [p.split(':')[0].replace('/', '') for p in get_open_positions()]
    signals_found = 0 
    
    for symbol in candidates:
        if signals_found >= TARGET_ACTIVE_COINS: break
        clean_candidate = symbol.split(':')[0].replace('/', '')
        if clean_candidate in current_positions: continue 
        if len(current_positions) >= 10: break

        if not analyze_potential(symbol): continue 
        if not ensure_model_exists(symbol): continue 
        if analyze_market(symbol, btc_trend):
            signals_found += 1
            time.sleep(1)

    print("-" * 50)
    print(f"[系統] 本輪結束。")

if __name__ == "__main__":
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(DATA_DIR, exist_ok=True)
    fund_manager_cycle()
    schedule.every(1).hours.do(fund_manager_cycle)
    while True:
        schedule.run_pending()
        time.sleep(1)