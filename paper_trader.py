import pandas as pd
import ccxt
import os
from datetime import datetime
import time

# --- 設定 ---
PORTFOLIO_FILE = 'data/paper_portfolio.csv'

def init_portfolio():
    """初始化帳本 CSV (新增 Strategy 與 Leverage 欄位)"""
    if not os.path.exists(PORTFOLIO_FILE):
        df = pd.DataFrame(columns=[
            'Symbol', 'Action', 'Strategy', 'Leverage',  # ✅ 新增欄位
            'Entry_Time', 'Entry_Price', 
            'TP', 'SL', 'Status', 'Exit_Time', 'Exit_Price', 'PnL_Percent'
        ])
        df.to_csv(PORTFOLIO_FILE, index=False)
        print(f"[SYSTEM] 建立新帳本: {PORTFOLIO_FILE}")

def get_open_positions():
    """回傳目前持有中的幣種清單"""
    init_portfolio()
    try:
        df = pd.read_csv(PORTFOLIO_FILE)
        if df.empty: return []
        return df[df['Status'] == 'OPEN']['Symbol'].tolist()
    except Exception as e:
        print(f"[ERROR] 讀取帳本失敗: {e}")
        return []

def execute_trade(symbol, action, price, tp, sl, strategy, leverage):
    """
    ✅ 執行開倉 (接收策略與槓桿參數)
    """
    init_portfolio()
    try:
        df = pd.read_csv(PORTFOLIO_FILE)
        
        if symbol in get_open_positions():
            print(f"[TRADE] 拒絕開倉: 已持有 {symbol} 部位。")
            return False

        new_trade = {
            'Symbol': symbol,
            'Action': action,
            'Strategy': strategy,   # ✅ 寫入策略
            'Leverage': leverage,   # ✅ 寫入槓桿
            'Entry_Time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'Entry_Price': price,
            'TP': tp,
            'SL': sl,
            'Status': 'OPEN',
            'Exit_Time': None,
            'Exit_Price': None,
            'PnL_Percent': 0.0
        }
        
        df = pd.concat([df, pd.DataFrame([new_trade])], ignore_index=True)
        df.to_csv(PORTFOLIO_FILE, index=False)
        
        print(f"[TRADE] 訂單已建立: {symbol} {action} [{strategy} | {leverage}x] @ {price:.4f}")
        return True
        
    except Exception as e:
        print(f"[ERROR] 下單失敗: {e}")
        return False

def monitor_positions():
    """監控持倉"""
    init_portfolio()
    try:
        df = pd.read_csv(PORTFOLIO_FILE)
        if df.empty: return

        # 強制轉型避免 Pandas 報錯
        df['Exit_Time'] = df['Exit_Time'].astype('object')
        df['Status'] = df['Status'].astype('object')

        open_trades = df[df['Status'] == 'OPEN']
        if open_trades.empty: return 

        print(f"[TRADER] 正在監控 {len(open_trades)} 筆持倉損益...")
        
        exchange = ccxt.binance()
        updated_count = 0
        
        for index, row in open_trades.iterrows():
            symbol = row['Symbol']
            entry_price = row['Entry_Price']
            tp = row['TP']
            sl = row['SL']
            action = row['Action']
            
            try:
                ticker = exchange.fetch_ticker(symbol)
                current_price = ticker['last']
                exit_reason = None
                
                # 計算原始漲跌幅 (不含槓桿)
                if action == 'BUY':
                    raw_pnl = (current_price - entry_price) / entry_price
                else:
                    raw_pnl = (entry_price - current_price) / entry_price
                
                # 這裡我們只記錄原始漲幅，顯示時再乘槓桿，保持數據純淨
                pnl_percent = raw_pnl * 100 

                # 檢查出場條件
                if action == 'BUY':
                    if current_price >= tp: exit_reason = 'WIN'
                    elif current_price <= sl: exit_reason = 'LOSS'
                elif action == 'SELL':
                    if current_price <= tp: exit_reason = 'WIN'
                    elif current_price >= sl: exit_reason = 'LOSS'
                
                if exit_reason:
                    df.at[index, 'Status'] = 'CLOSED'
                    df.at[index, 'Exit_Time'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    df.at[index, 'Exit_Price'] = float(current_price)
                    df.at[index, 'PnL_Percent'] = round(pnl_percent, 2)
                    
                    log_tag = "[WIN] 🎉" if pnl_percent > 0 else "[LOSS] 😭"
                    print(f"   {log_tag} {symbol} 平倉！ PnL: {pnl_percent:.2f}%")
                    updated_count += 1
                else:
                    print(f"   [HOLD] {symbol}: {pnl_percent:.2f}%")

            except Exception as e:
                print(f"[ERROR] 查詢 {symbol} 價格失敗: {e}")
                
        if updated_count > 0:
            df.to_csv(PORTFOLIO_FILE, index=False)
            print(f"[TRADER] 已更新 {updated_count} 筆交易紀錄。")

    except Exception as e:
        print(f"[ERROR] 監控持倉時發生錯誤: {e}")

if __name__ == "__main__":
    monitor_positions()