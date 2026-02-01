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

# ✅ 設定 Pandas 顯示格式
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

    # 1. 讀取數據
    df = pd.read_csv(PORTFOLIO_FILE)
    if df.empty:
        print("[INFO] 尚無交易紀錄。")
        return
        
    # 補上預設值 (防止舊版 CSV 報錯)
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
        
        leverage = float(row.get('Leverage', 1))
        real_pnl_percent = row['PnL_Percent'] * leverage
        
        profit_loss_usd = bet_size * (real_pnl_percent / 100)
        balance += profit_loss_usd
        
        report_data.append({
            '時間': row['Exit_Time'].strftime('%Y-%m-%d %H:%M'),
            '幣種': row['Symbol'],
            '方向': row['Action'],
            '策略': row['Strategy'],
            '槓桿': f"{leverage:.0f}x",
            '狀態': '已平倉',
            '進場價': row['Entry_Price'],
            '現價/出場': row['Exit_Price'],
            '報酬率%': real_pnl_percent,
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
            
            if action == 'BUY':
                raw_pct = (curr_price - entry_price) / entry_price * 100
            else:
                raw_pct = (entry_price - curr_price) / entry_price * 100
            
            real_pnl_percent = raw_pct * leverage
            bet_size = realized_balance * POSITION_SIZE_RATIO
            float_pnl_usd = bet_size * (real_pnl_percent / 100)
            
            floating_pnl_total += float_pnl_usd
            
            report_data.append({
                '時間': '持倉中',
                '幣種': symbol,
                '方向': action,
                '策略': row['Strategy'],
                '槓桿': f"{leverage:.0f}x",
                '狀態': '持倉中',
                '進場價': entry_price,
                '現價/出場': curr_price,
                '報酬率%': real_pnl_percent,
                '損益(USD)': float_pnl_usd,
                '結算餘額': realized_balance + floating_pnl_total
            })

    final_equity = realized_balance + floating_pnl_total
    
    # --- 輸出報表 ---
    report_df = pd.DataFrame(report_data)
    
    print("\n" + "="*120)
    print(f"💰 資產績效報表 (本金: ${INITIAL_CAPITAL:,.0f})")
    print("="*120)
    
    if not report_df.empty:
        display_cols = ['時間', '幣種', '方向', '策略', '槓桿', '狀態', '進場價', '現價/出場', '報酬率%', '損益(USD)', '結算餘額']
        print(report_df[display_cols].to_string(index=False))
    else:
        print("目前無任何交易紀錄。")

    print("-" * 120)
    
    # =========================================================
    # ✅ [新增] 勝率與交易統計計算
    # =========================================================
    total_closed = len(closed_df)
    winning_trades = len(closed_df[closed_df['PnL_Percent'] > 0])
    losing_trades = len(closed_df[closed_df['PnL_Percent'] <= 0])
    
    # 防止除以零錯誤
    win_rate = (winning_trades / total_closed * 100) if total_closed > 0 else 0.0
    
    total_return_pct = ((final_equity - INITIAL_CAPITAL) / INITIAL_CAPITAL) * 100
    
    # 顯示統計資訊
    print(f"[SUMMARY] 交易筆數 (Trades) : {total_closed} (Win: {winning_trades} / Loss: {losing_trades})")
    print(f"[SUMMARY] 勝率 (Win Rate)   : {win_rate:.2f}%")  # ✅ 顯示勝率
    print(f"[SUMMARY] 現金餘額 (Balance): ${realized_balance:,.2f}")
    print(f"[SUMMARY] 浮動損益 (Floating): ${floating_pnl_total:,.2f}")
    print(f"[SUMMARY] 帳戶淨值 (Equity) : ${final_equity:,.2f}")
    print(f"[SUMMARY] 總報酬率 (ROI)    : {total_return_pct:.2f}%")
    print("=" * 120)

    # --- 繪圖 ---
    try:
        # 繪圖邏輯不變
        balance = INITIAL_CAPITAL
        equity_curve = [INITIAL_CAPITAL]
        dates = [closed_df['Exit_Time'].min() - pd.Timedelta(hours=4)] if not closed_df.empty else [datetime.now()]
        
        # 重算一次只為了畫圖 (使用 closed_df)
        temp_bal = INITIAL_CAPITAL
        for index, row in closed_df.iterrows():
            bet_size = temp_bal * POSITION_SIZE_RATIO
            leverage = float(row.get('Leverage', 1))
            pnl = row['PnL_Percent'] * leverage
            temp_bal += bet_size * (pnl / 100)
            dates.append(row['Exit_Time'])
            equity_curve.append(temp_bal)
            
        # 加入浮動損益點
        dates.append(datetime.now())
        equity_curve.append(final_equity)

        plt.figure(figsize=(12, 6))
        plt.plot(dates[:-1], equity_curve[:-1], marker='o', linestyle='-', color='#1f77b4', label='Realized Balance')
        plt.plot(dates[-2:], equity_curve[-2:], marker='d', linestyle='--', color='orange', label='Floating Equity')
        plt.axhline(y=INITIAL_CAPITAL, color='gray', linestyle='--', alpha=0.5, label='Initial Capital')
        
        plt.title(f"Portfolio Performance (Win Rate: {win_rate:.1f}% | ROI: {total_return_pct:.2f}%)", fontsize=14)
        plt.xlabel("Date")
        plt.ylabel("Value (USD)")
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))
        plt.gcf().autofmt_xdate()

        output_img = 'data/performance_chart.png'
        plt.savefig(output_img)
        print(f"[SUCCESS] 損益圖表已儲存為: {output_img}")
    except Exception as e:
        print(f"[ERROR] 繪圖失敗: {e}")

if __name__ == "__main__":
    generate_report()