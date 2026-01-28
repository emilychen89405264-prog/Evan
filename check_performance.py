import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import ccxt
import os
from datetime import datetime

# --- 設定 ---
PORTFOLIO_FILE = 'data/paper_portfolio.csv'
INITIAL_CAPITAL = 50000.0   
POSITION_SIZE_RATIO = 0.1   

# ✅ 關鍵設定：解決中文對齊問題
pd.set_option('display.unicode.east_asian_width', True)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)
pd.set_option('display.float_format', '{:,.2f}'.format)

def get_current_prices(symbols):
    prices = {}
    if not symbols: return prices
    print(f"[NET] 查詢 {len(symbols)} 個持倉現價...")
    try:
        exchange = ccxt.binance()
        tickers = exchange.fetch_tickers(symbols)
        for symbol, ticker in tickers.items():
            prices[symbol] = ticker['last']
    except Exception as e:
        print(f"[ERROR] 價格查詢失敗: {e}")
    return prices

def generate_report():
    if not os.path.exists(PORTFOLIO_FILE):
        print("[ERROR] 找不到交易紀錄檔。")
        return

    # 1. 讀取數據 (處理新增的欄位缺失值，防止舊資料報錯)
    df = pd.read_csv(PORTFOLIO_FILE)
    if df.empty:
        print("[INFO] 尚無交易紀錄。")
        return
        
    # 如果舊資料沒有 Strategy/Leverage 欄位，補上預設值
    if 'Strategy' not in df.columns: df['Strategy'] = 'N/A'
    if 'Leverage' not in df.columns: df['Leverage'] = 1

    closed_df = df[df['Status'] == 'CLOSED'].copy()
    open_df = df[df['Status'] == 'OPEN'].copy()
    
    closed_df['Exit_Time'] = pd.to_datetime(closed_df['Exit_Time'])
    closed_df = closed_df.sort_values(by='Exit_Time')

    # --- 計算已實現損益 ---
    balance = INITIAL_CAPITAL
    report_data = []

    for index, row in closed_df.iterrows():
        bet_size = balance * POSITION_SIZE_RATIO
        
        # ✅ 計算真實損益 (原始漲跌幅 * 槓桿)
        leverage = float(row.get('Leverage', 1))
        real_pnl_percent = row['PnL_Percent'] * leverage
        
        profit_loss_usd = bet_size * (real_pnl_percent / 100)
        balance += profit_loss_usd
        
        report_data.append({
            '時間': row['Exit_Time'].strftime('%Y-%m-%d %H:%M'),
            '幣種': row['Symbol'],
            '方向': row['Action'],
            '策略': row['Strategy'],  # ✅ 顯示策略
            '槓桿': f"{leverage:.0f}x",   # ✅ 顯示槓桿
            '狀態': '已平倉',
            '進場價': row['Entry_Price'],
            '現價/出場': row['Exit_Price'],
            '報酬率%': real_pnl_percent, # 顯示槓桿後報酬
            '損益(USD)': profit_loss_usd,
            '結算餘額': balance
        })

    realized_balance = balance 

    # --- 計算持倉浮動損益 ---
    floating_pnl_total = 0
    
    if not open_df.empty:
        open_symbols = open_df['Symbol'].unique().tolist()
        current_prices = get_current_prices(open_symbols)
        
        for index, row in open_df.iterrows():
            symbol = row['Symbol']
            entry_price = row['Entry_Price']
            action = row['Action']
            leverage = float(row.get('Leverage', 1))
            
            curr_price = current_prices.get(symbol, entry_price)
            
            # 計算原始漲跌
            if action == 'BUY':
                raw_pct = (curr_price - entry_price) / entry_price * 100
            else:
                raw_pct = (entry_price - curr_price) / entry_price * 100
            
            # ✅ 乘上槓桿
            real_pnl_percent = raw_pct * leverage
            
            bet_size = realized_balance * POSITION_SIZE_RATIO
            float_pnl_usd = bet_size * (real_pnl_percent / 100)
            
            floating_pnl_total += float_pnl_usd
            
            report_data.append({
                '時間': '持倉中',
                '幣種': symbol,
                '方向': action,
                '策略': row['Strategy'],  # ✅ 顯示策略
                '槓桿': f"{leverage:.0f}x",   # ✅ 顯示槓桿
                '狀態': '持倉中',
                '進場價': entry_price,
                '現價/出場': curr_price,
                '報酬率%': real_pnl_percent, # 顯示槓桿後報酬
                '損益(USD)': float_pnl_usd,
                '結算餘額': realized_balance + floating_pnl_total
            })

    final_equity = realized_balance + floating_pnl_total
    
    # --- 輸出報表 ---
    report_df = pd.DataFrame(report_data)
    
    print("\n" + "="*120) # 加長分隔線以容納新欄位
    print(f"💰 資產績效報表 (本金: ${INITIAL_CAPITAL:,.0f})")
    print("="*120)
    
    if not report_df.empty:
        # ✅ 這裡定義你想看到的欄位順序
        display_cols = ['時間', '幣種', '方向', '策略', '槓桿', '狀態', '進場價', '現價/出場', '報酬率%', '損益(USD)', '結算餘額']
        print(report_df[display_cols].to_string(index=False))
    else:
        print("目前無任何交易紀錄。")

    print("-" * 120)
    
    total_return_pct = ((final_equity - INITIAL_CAPITAL) / INITIAL_CAPITAL) * 100
    
    print(f"[SUMMARY] 現金餘額 (Balance): ${realized_balance:,.2f}")
    print(f"[SUMMARY] 浮動損益 (Floating): ${floating_pnl_total:,.2f}")
    print(f"[SUMMARY] 帳戶淨值 (Equity) : ${final_equity:,.2f}")
    print(f"[SUMMARY] 總報酬率 (ROI)    : {total_return_pct:.2f}%")
    print("=" * 120)

if __name__ == "__main__":
    generate_report()