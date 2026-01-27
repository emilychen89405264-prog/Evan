import pandas as pd
import ccxt
import os
from datetime import datetime
import time

# --- 設定 ---
PORTFOLIO_FILE = 'data/paper_portfolio.csv'

def init_portfolio():
    """初始化帳本 CSV"""
    if not os.path.exists(PORTFOLIO_FILE):
        df = pd.DataFrame(columns=[
            'Symbol', 'Action', 'Entry_Time', 'Entry_Price', 
            'TP', 'SL', 'Status', 'Exit_Time', 'Exit_Price', 'PnL_Percent'
        ])
        df.to_csv(PORTFOLIO_FILE, index=False)
        print(f"[SYSTEM] 建立新帳本: {PORTFOLIO_FILE}")

def get_open_positions():
    """回傳目前持有中的幣種清單 (List)"""
    init_portfolio()
    try:
        df = pd.read_csv(PORTFOLIO_FILE)
        # 篩選 Status 為 OPEN 的幣種
        open_pos = df[df['Status'] == 'OPEN']['Symbol'].tolist()
        return open_pos
    except Exception as e:
        print(f"[ERROR] 讀取帳本失敗: {e}")
        return []

def execute_trade(symbol, action, price, tp, sl):
    """執行開倉 (寫入帳本)"""
    init_portfolio()
    try:
        df = pd.read_csv(PORTFOLIO_FILE)
        
        # 檢查是否已經持有該幣種 (避免重複下單)
        if symbol in get_open_positions():
            print(f"[TRADE] 拒絕開倉: 已持有 {symbol} 部位。")
            return False

        new_trade = {
            'Symbol': symbol,
            'Action': action,
            'Entry_Time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'Entry_Price': price,
            'TP': tp,
            'SL': sl,
            'Status': 'OPEN',
            'Exit_Time': None,
            'Exit_Price': None,
            'PnL_Percent': 0.0
        }
        
        # 使用 concat 替代 append (新版 pandas 寫法)
        df = pd.concat([df, pd.DataFrame([new_trade])], ignore_index=True)
        df.to_csv(PORTFOLIO_FILE, index=False)
        
        print(f"[TRADE] 訂單已建立: {symbol} {action} @ {price:.4f} (TP:{tp:.4f}/SL:{sl:.4f})")
        return True
        
    except Exception as e:
        print(f"[ERROR] 下單失敗: {e}")
        return False

def monitor_positions():
    """
    監控所有持倉：檢查是否觸發 TP 或 SL
    """
    init_portfolio()
    try:
        df = pd.read_csv(PORTFOLIO_FILE)
        
        # 只看 OPEN 的單
        open_trades = df[df['Status'] == 'OPEN']
        if open_trades.empty:
            return # 手上沒單，沒事做

        print(f"[TRADER] 正在監控 {len(open_trades)} 筆持倉損益...")
        
        # 連線交易所查最新價格
        exchange = ccxt.binance()
        
        updated_count = 0
        
        for index, row in open_trades.iterrows():
            symbol = row['Symbol']
            action = row['Action']
            entry_price = row['Entry_Price']
            tp = row['TP']
            sl = row['SL']
            
            try:
                # 抓取該幣種最新價格
                ticker = exchange.fetch_ticker(symbol)
                current_price = ticker['last']
                
                exit_reason = None
                pnl = 0.0
                
                # --- 判斷多單 (BUY) ---
                if action == 'BUY':
                    if current_price >= tp:
                        exit_reason = 'WIN'
                        pnl = (current_price - entry_price) / entry_price * 100
                    elif current_price <= sl:
                        exit_reason = 'LOSS'
                        pnl = (current_price - entry_price) / entry_price * 100 # 負數
                
                # --- 判斷空單 (SELL) ---
                elif action == 'SELL':
                    if current_price <= tp:
                        exit_reason = 'WIN'
                        pnl = (entry_price - current_price) / entry_price * 100
                    elif current_price >= sl:
                        exit_reason = 'LOSS'
                        pnl = (entry_price - current_price) / entry_price * 100 # 負數
                
                # 如果觸發出場條件，更新帳本
                if exit_reason:
                    df.at[index, 'Status'] = 'CLOSED'
                    df.at[index, 'Exit_Time'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    df.at[index, 'Exit_Price'] = current_price
                    df.at[index, 'PnL_Percent'] = round(pnl, 2)
                    
                    log_tag = "[WIN] 🎉" if pnl > 0 else "[LOSS] 😭"
                    print(f"   {log_tag} {symbol} 平倉！ PnL: {pnl:.2f}% (現價: {current_price})")
                    updated_count += 1
                else:
                    # 尚未平倉，顯示即時浮動損益
                    if action == 'BUY':
                        float_pnl = (current_price - entry_price) / entry_price * 100
                    else:
                        float_pnl = (entry_price - current_price) / entry_price * 100
                    print(f"   [HOLD] {symbol}: {float_pnl:.2f}%")

            except Exception as e:
                print(f"[ERROR] 查詢 {symbol} 價格失敗: {e}")
                
        # 如果有任何變動，才寫入硬碟
        if updated_count > 0:
            df.to_csv(PORTFOLIO_FILE, index=False)
            print(f"[TRADER] 已更新 {updated_count} 筆交易紀錄。")

    except Exception as e:
        print(f"[ERROR] 監控持倉時發生錯誤: {e}")

# 獨立測試用
if __name__ == "__main__":
    # 測試開一單
    # execute_trade('ETH/USDT', 'BUY', 2000, 2100, 1900)
    monitor_positions()