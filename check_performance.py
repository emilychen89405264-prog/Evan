import pandas as pd
import ccxt
import time
import os
import matplotlib
import matplotlib.pyplot as plt
from datetime import datetime


matplotlib.use('Agg')


CSV_FILE = 'data/paper_portfolio.csv' 
INITIAL_BALANCE = 50000
POSITION_SIZE_RATIO = 0.1
REFRESH_SECONDS = 60 
CHART_FILE = 'equity_curve.png'


pd.set_option('display.unicode.east_asian_width', True)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 2000)

binance = ccxt.binance({
    'options': {'defaultType': 'future'}
})
binance.set_sandbox_mode(False)
binance.urls['api']['fapiPublic'] = 'https://fapi.binance.com/fapi/v1'
binance.urls['api']['fapiPrivate'] = 'https://fapi.binance.com/fapi/v1'

def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')

def get_current_price(symbol):
    try:
        clean_symbol = symbol.split(':')[0].replace('/', '')
        ticker = binance.fetch_ticker(clean_symbol)
        return float(ticker['last'])
    except Exception as e:
        return None

def generate_equity_curve(df):
    try:
        chart_df = df.copy()
        chart_df = chart_df.sort_values(by='raw_time')
        
        # 計算淨值
        chart_df['equity'] = INITIAL_BALANCE + chart_df['損益(U)'].cumsum()
        
        # 準備繪圖數據
        dates = chart_df['raw_time']
        equity = chart_df['equity']
        
        # 設定畫布
        plt.figure(figsize=(12, 6))
        
        # 1. 畫折線
        plt.plot(dates, equity, color='#333333', linewidth=1.5, label='Total Equity')
        
        # 2. 畫基準線 (本金)
        plt.axhline(y=INITIAL_BALANCE, color='gray', linestyle='--', linewidth=1, label='Initial Balance')

        # 3. 填充顏色 (綠/紅)
        plt.fill_between(dates, equity, INITIAL_BALANCE, 
                         where=(equity >= INITIAL_BALANCE), 
                         interpolate=True, color='green', alpha=0.2)
        
        plt.fill_between(dates, equity, INITIAL_BALANCE, 
                         where=(equity < INITIAL_BALANCE), 
                         interpolate=True, color='red', alpha=0.2)

        # 設定標題與標籤
        plt.title(f'Paper Trading Equity Curve (Initial: ${INITIAL_BALANCE})', fontsize=14)
        plt.xlabel('Date', fontsize=10)
        plt.ylabel('Balance (USDT)', fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        # 存檔
        plt.savefig(CHART_FILE)
        plt.close()
        return True
    except Exception as e:
        print(f"[繪圖錯誤] {e}")
        return False

def check_performance():
    if not os.path.exists(CSV_FILE):
        print(f"[等待] 尚未找到交易紀錄檔: {CSV_FILE}")
        return

    try:
        df = pd.read_csv(CSV_FILE)
        df.columns = df.columns.str.strip()
    except Exception as e:
        print(f"[錯誤] 讀取 CSV 失敗: {e}")
        return

    processed_data = []

    for index, row in df.iterrows():
        try:
            symbol = row.get('Symbol') or row.get('symbol')
            action = row.get('Action') or row.get('action')
            status = row.get('Status') or row.get('status')
            entry_price = float(row.get('Entry_Price') or 0)
            leverage = float(row.get('Leverage') or 1)
            tp_price = float(row.get('TP') or row.get('tp') or 0)
            sl_price = float(row.get('SL') or row.get('sl') or 0)
            entry_time_str = row.get('Entry_Time') or row.get('timestamp')
            
            try:
                entry_time_dt = pd.to_datetime(entry_time_str)
                time_display = str(entry_time_dt)[5:16]
            except:
                time_display = "時間錯誤"
                entry_time_dt = datetime.now()

            position_usdt = INITIAL_BALANCE * POSITION_SIZE_RATIO 
            size = position_usdt / entry_price if entry_price > 0 else 0
            
            pnl_u = 0
            roi = 0.0
            current_price = entry_price
            note = "未知"
            sort_group = 0 
            display_current_price = f"{entry_price:.4f}"

            if status == 'CLOSED':
                note = "已平倉"
                sort_group = 0
                exit_price = float(row.get('Exit_Price') or entry_price)
                current_price = exit_price
                display_current_price = f"{exit_price:.4f}"
                
                if 'PnL_Percent' in row and pd.notnull(row['PnL_Percent']):
                    roi = float(row['PnL_Percent'])
                    pnl_u = position_usdt * (roi / 100)
                else:
                    if action == 'BUY': pnl_u = (exit_price - entry_price) * size
                    else: pnl_u = (entry_price - exit_price) * size
                    roi = (pnl_u / position_usdt) * 100

            elif status == 'OPEN' or status == 'Held':
                note = "持倉中"
                sort_group = 1
                live_price = get_current_price(symbol)
                
                if live_price:
                    current_price = live_price
                    display_current_price = f"{live_price:.4f}"
                    
                    if action == 'BUY':
                        raw_pnl = (current_price - entry_price) * size
                    else: 
                        raw_pnl = (entry_price - current_price) * size
                    
                    pnl_u = raw_pnl
                    roi = (raw_pnl / position_usdt) * 100
                else:
                    note = "連線失敗"
                    display_current_price = "❌"

            processed_data.append({
                "raw_time": entry_time_dt,
                "sort_group": sort_group,
                "時間": time_display,
                "幣種": symbol,
                "方向": action,
                "槓桿": leverage, 
                "狀態": note,
                "進場價": entry_price,
                "現價/出場": display_current_price,
                "TP": tp_price,
                "SL": sl_price,
                "報酬率%": roi,      
                "損益(U)": pnl_u     
            })
        except: continue

    res_df = pd.DataFrame(processed_data)
    if res_df.empty:
        print("尚無交易資料")
        return

    # 🔥 生成帶填充色的圖表
    chart_generated = generate_equity_curve(res_df)

    # 排序與計算
    res_df = res_df.sort_values(by=['sort_group', 'raw_time'])
    res_df['balance_calc'] = INITIAL_BALANCE + res_df['損益(U)'].cumsum()

    closed_trades = res_df[res_df['sort_group'] == 0]
    total_closed = len(closed_trades)
    win_count = len(closed_trades[closed_trades['損益(U)'] > 0])
    loss_count = len(closed_trades[closed_trades['損益(U)'] <= 0])
    win_rate = (win_count / total_closed * 100) if total_closed > 0 else 0.0
    
    total_realized = closed_trades['損益(U)'].sum()
    total_unrealized = res_df[res_df['sort_group'] == 1]['損益(U)'].sum()
    final_equity = INITIAL_BALANCE + total_realized + total_unrealized
    total_roi = ((final_equity - INITIAL_BALANCE) / INITIAL_BALANCE) * 100

    # 格式化輸出
    final_df = pd.DataFrame()
    final_df['時間'] = res_df['時間']
    final_df['幣種'] = res_df['幣種']
    final_df['方向'] = res_df['方向']
    
    final_df['槓桿'] = res_df['槓桿'].apply(lambda x: f"{x:.0f}x")
    final_df['TP'] = res_df['TP'].apply(lambda x: f"{x:,.4f}" if x > 0 else "-")
    final_df['SL'] = res_df['SL'].apply(lambda x: f"{x:,.4f}" if x > 0 else "-")
    final_df['進場價'] = res_df['進場價'].apply(lambda x: f"{x:,.4f}")
    final_df['現價/出場'] = res_df['現價/出場']
    
    # ✅ 改回純文字格式化，移除顏色代碼
    final_df['報酬率%'] = res_df['報酬率%'].apply(lambda x: f"{x:+.2f}%" if x != 0 else "---")
    final_df['損益(U)'] = res_df['損益(U)'].apply(lambda x: f"{x:+.2f}" if x != 0 else "---")
    
    final_df['累計餘額'] = res_df['balance_calc'].apply(lambda x: f"{x:,.2f}")

    print(f"\n{'='*155}")
    print(f"資產績效報表 (Binance Mainnet) - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*155}\n")
    
    print(final_df.to_string(index=False, col_space=14, justify='right'))
    
    print(f"\n{'-'*155}")
    print(f"交易統計 (Stats)        : 總筆數 {total_closed} (勝 {win_count} / 負 {loss_count})")
    print(f"勝率 (Win Rate)         : {win_rate:.2f}%")
    print("-" * 155)
    
    print(f"已實現損益 (Realized)   : ${total_realized:,.2f}")
    print(f"浮動損益 (Floating)     : ${total_unrealized:,.2f}")
    print(f"當前總淨值 (Total Equity): ${final_equity:,.2f}")
    print(f"總報酬率 (Total ROI)    : {total_roi:+.2f}%")
    
    if chart_generated:
        print(f"收益曲線圖已更新: {CHART_FILE}")
        
    print(f"{'='*155}")
    print(f"\n[系統] 監控中... 下次更新於 {REFRESH_SECONDS} 秒後 (按 Ctrl+C 停止)")

if __name__ == "__main__":
    try:
        while True:
            clear_screen()
            check_performance()
            time.sleep(REFRESH_SECONDS)
    except KeyboardInterrupt:
        print("\n\n[系統] 監控已手動停止。")