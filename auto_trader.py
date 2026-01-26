import time
import pandas as pd
import numpy as np
import schedule
import ccxt
import joblib
from datetime import datetime
from tensorflow.keras.models import load_model
from predict_optimized import get_latest_data, process_data, FEATURE_COLS

# --- 核心參數設定 ---
SYMBOL = 'ETH/USDT'
TIMEFRAME = '4h'
LOG_FILE = 'data/paper_trading_log.csv'
MODEL_PATH = 'models/best_crypto_model.keras'
SCALER_PATH = 'models/scaler.pkl'

CONFIDENCE_THRESHOLD = 37.0
MAX_OPEN_POSITIONS = 3 

# --- 全域變數 (用來放只需載入一次的資源) ---
bot_model = None

def init_bot():
    """程式啟動時執行一次：載入模型與資源"""
    global bot_model
    print("[INIT] 正在初始化機器人資源...")
    try:
        # 在這裡載入模型，而不是在迴圈裡
        bot_model = load_model(MODEL_PATH)
        print("[SUCCESS] AI 模型載入成功！(由 GPU/CPU 託管中)")
    except Exception as e:
        print(f"[ERROR] 模型載入失敗: {e}")
        exit()

def check_opposite_direction(new_action, current_positions):
    """強制執行「多空交替」策略"""
    if current_positions == 0:
        return True

    try:
        df = pd.read_csv(LOG_FILE)
        if len(df) == 0:
            return True
        last_action = df.iloc[-1]['Action']
        
        if new_action == last_action:
            print(f"[STRATEGY] 策略阻擋：上一筆是 {last_action}，禁止連續同方向加碼。")
            return False
        return True
    except Exception as e:
        print(f"[WARNING] 讀取歷史紀錄發生錯誤: {e}")
        return False

def manage_positions(current_price, current_time):
    """檢查平倉邏輯"""
    try:
        df = pd.read_csv(LOG_FILE)
    except FileNotFoundError:
        return 0 

    pending_trades = df[df['Result'] == 'Pending']
    open_count = len(pending_trades)
    
    if open_count == 0:
        return 0

    print(f"[MONITOR] 正在監控 {open_count} 張持倉訂單...")
    trades_closed = 0

    for index, row in pending_trades.iterrows():
        entry_price = row['Entry_Price']
        tp = row['Take_Profit']
        sl = row['Stop_Loss']
        action = row['Action']
        
        result = 'Pending'
        pnl_percent = 0.0
        
        if action == 'BUY':
            if current_price >= tp:
                result = 'Win'
                pnl_percent = (tp - entry_price) / entry_price * 100
                print(f"   [WIN] BUY 訂單止盈出場！獲利: {pnl_percent:.2f}%")
            elif current_price <= sl:
                result = 'Loss'
                pnl_percent = (sl - entry_price) / entry_price * 100
                print(f"   [LOSS] BUY 訂單止損出場。虧損: {pnl_percent:.2f}%")
                
        elif action == 'SELL':
            if current_price <= tp:
                result = 'Win'
                pnl_percent = (entry_price - tp) / entry_price * 100
                print(f"   [WIN] SELL 訂單止盈出場！獲利: {pnl_percent:.2f}%")
            elif current_price >= sl:
                result = 'Loss'
                pnl_percent = (entry_price - sl) / entry_price * 100
                print(f"   [LOSS] SELL 訂單止損出場。虧損: {pnl_percent:.2f}%")

        if result != 'Pending':
            df.at[index, 'Result'] = result
            df.at[index, 'PnL_Percent'] = round(pnl_percent, 2)
            df.at[index, 'Exit_Time'] = current_time
            trades_closed += 1

    if trades_closed > 0:
        df.to_csv(LOG_FILE, index=False)
        print(f"[INFO] 已更新交易紀錄，平倉 {trades_closed} 筆。")
        
    return open_count - trades_closed

def open_new_trade(action, price, sl, tp, confidence, time_str):
    """寫入新訂單"""
    log_entry = {
        'Timestamp': time_str,
        'Action': action,
        'Entry_Price': price,
        'Stop_Loss': sl,
        'Take_Profit': tp,
        'Confidence': confidence,
        'Result': 'Pending',
        'PnL_Percent': 0.0,
        'Exit_Time': ''
    }
    
    try:
        df = pd.read_csv(LOG_FILE)
        df = pd.concat([df, pd.DataFrame([log_entry])], ignore_index=True)
    except FileNotFoundError:
        df = pd.DataFrame([log_entry])
        
    df.to_csv(LOG_FILE, index=False)
    print(f"[ORDER] 新訂單已建立：{action} @ {price}")

def bot_cycle():
    """機器人的完整思考循環"""
    global bot_model
    
    now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(f"\n[WAKEUP] {now} - 機器人喚醒中...")
    
    try:
        # 1. 獲取最新行情
        df_raw = get_latest_data() 
        X_input, last_candle = process_data(df_raw)
        
        current_price = last_candle['close']
        current_atr = last_candle['ATR']
        
        print(f"[MARKET] 當前市價: {current_price:.2f} | ATR: {current_atr:.2f}")

        # 2. 部位管理
        current_positions = manage_positions(current_price, now)
        
        # 3. 判斷持倉上限
        if current_positions >= MAX_OPEN_POSITIONS:
            print(f"[LIMIT] 目前持有 {current_positions}/{MAX_OPEN_POSITIONS} 張單，達上限。暫停開新單。")
            return 

        # 4. AI 預測
        print("[AI] 正在分析進場機會...")
        
        # 直接使用變數 bot_model
        prediction = bot_model.predict(X_input, verbose=0)
        
        prob_buy  = prediction[0][1] * 100
        prob_sell = prediction[0][2] * 100
        action_idx = np.argmax(prediction)

        print(f"   [PREDICT] 預測機率: Buy({prob_buy:.1f}%) | Sell({prob_sell:.1f}%)")

        # 5. 執行下單邏輯
        new_action = None
        if action_idx == 1 and prob_buy > CONFIDENCE_THRESHOLD:
            new_action = "BUY"
        elif action_idx == 2 and prob_sell > CONFIDENCE_THRESHOLD:
            new_action = "SELL"
            
        if new_action:
            if check_opposite_direction(new_action, current_positions):
                if new_action == "BUY":
                    tp = current_price + (current_atr * 2)
                    sl = current_price - (current_atr * 1)
                else:
                    tp = current_price - (current_atr * 2)
                    sl = current_price + (current_atr * 1)
                open_new_trade(new_action, current_price, sl, tp, max(prob_buy, prob_sell), now)
            else:
                print(f"   [WAIT] 訊號 {new_action} 被策略濾網攔截。")
        else:
            print("   [WAIT] 訊號未達標準，繼續觀望。")

    except Exception as e:
        print(f"[ERROR] 發生錯誤: {e}")

if __name__ == "__main__":
    print(f"[SYSTEM] 雙向對沖機器人啟動！(Max Positions: {MAX_OPEN_POSITIONS})")
    
    # 初始化
    init_bot()
    
    # 測試執行
    bot_cycle()
    
    # 排程設定
    schedule.every(3).minutes.do(bot_cycle) 
    
    while True:
        schedule.run_pending()
        time.sleep(1)