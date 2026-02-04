import pandas as pd
import ccxt
import time
import os

# ==========================================
# 🛠️ 設定區
# ==========================================
CSV_FILE = 'data/paper_portfolio.csv'
INITIAL_BALANCE = 50000
POSITION_SIZE_RATIO = 0.1

# ✅ 設定 Pandas 顯示參數
pd.set_option('display.unicode.east_asian_width', True)  # 支援中文寬度
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 2000)                     # 設定總寬度大一點

# 初始化查價引擎
bybit = ccxt.bybit({'options': {'defaultType': 'linear'}})
binance = ccxt.binance()

def get_price_with_fallback(symbol):
    """ 雙重查價機制 """
    try:
        ticker = bybit.fetch_ticker(symbol)
        return float(ticker['last']), "Bybit"
    except:
        pass
    try:
        clean_symbol = symbol.split(':')[0].replace('/', '')
        binance_symbol = symbol.split(':')[0]
        ticker = binance.fetch_ticker(binance_symbol)
        return float(ticker['last']), "Binance"
    except:
        pass
    return None, "Fail"

def check_performance():
    if not os.path.exists(CSV_FILE):
        print(f"[錯誤] 找不到交易紀錄檔: {CSV_FILE}")
        return

    print(">>> 正在讀取並計算最新損益 (請稍候)...")
    
    try:
        df = pd.read_csv(CSV_FILE)
        df.columns = df.columns.str.strip() # 去除欄位空白
    except Exception as e:
        print(f"[錯誤] 讀取 CSV 失敗: {e}")
        return

    # 準備列表存資料
    processed_data = []

    # 1. 第一輪迴圈：計算所有單子的損益
    for index, row in df.iterrows():
        try:
            symbol = row.get('Symbol') or row.get('symbol')
            action = row.get('Action') or row.get('action')
            status = row.get('Status') or row.get('status')
            entry_price = float(row.get('Entry_Price') or row.get('entry_price') or 0)
            
            entry_time_str = row.get('Entry_Time') or row.get('timestamp')
            entry_time_dt = pd.to_datetime(entry_time_str)
            time_display = str(entry_time_dt)[5:16]

            # 計算倉位大小 (模擬)
            position_usdt = INITIAL_BALANCE * POSITION_SIZE_RATIO 
            size = position_usdt / entry_price if entry_price > 0 else 0
            
            pnl_u = 0
            roi = 0.0
            current_price = entry_price
            note = "未知"
            sort_group = 0 # 0:已平倉, 1:持倉中

            # --- 情況 A: 已平倉 ---
            if status == 'CLOSED':
                note = "已平倉"
                sort_group = 0
                exit_price = float(row.get('Exit_Price') or row.get('exit_price') or entry_price)
                current_price = exit_price
                
                if 'PnL_Percent' in row and pd.notnull(row['PnL_Percent']):
                    roi = float(row['PnL_Percent'])
                    pnl_u = position_usdt * (roi / 100)
                else:
                    if action == 'BUY':
                        pnl_u = (exit_price - entry_price) * size
                    else:
                        pnl_u = (entry_price - exit_price) * size
                    roi = (pnl_u / position_usdt) * 100

            # --- 情況 B: 持倉中 ---
            elif status == 'OPEN' or status == 'Held':
                note = "持倉中"
                sort_group = 1
                live_price, source = get_price_with_fallback(symbol)
                
                if live_price:
                    current_price = live_price
                    if action == 'BUY':
                        raw_pnl = (current_price - entry_price) * size
                    else: # SELL
                        raw_pnl = (entry_price - current_price) * size
                    
                    pnl_u = raw_pnl
                    roi = (raw_pnl / position_usdt) * 100
                else:
                    note = "查無價"

            # 存入列表
            processed_data.append({
                "raw_time": entry_time_dt,
                "sort_group": sort_group,
                "時間": time_display,
                "幣種": symbol,
                "方向": action,
                "狀態": note,
                "進場價": entry_price,
                "現價/出場": current_price,
                "報酬率%": roi,      
                "損益(U)": pnl_u     
            })
            
        except Exception as e:
            continue

    # 2. 轉成 DataFrame 進行排序與計算
    res_df = pd.DataFrame(processed_data)
    
    if res_df.empty:
        print("尚無有效交易資料")
        return

    # 排序：先已平倉(0)->持倉中(1)，內部再按時間
    res_df = res_df.sort_values(by=['sort_group', 'raw_time'])

    # 計算累計餘額
    res_df['balance_calc'] = INITIAL_BALANCE + res_df['損益(U)'].cumsum()

    # --- 📊 統計數據計算 (新增部分) ---
    closed_trades = res_df[res_df['sort_group'] == 0]
    total_closed = len(closed_trades)
    
    win_count = len(closed_trades[closed_trades['損益(U)'] > 0])
    loss_count = len(closed_trades[closed_trades['損益(U)'] <= 0])
    
    win_rate = (win_count / total_closed * 100) if total_closed > 0 else 0.0

    total_realized = closed_trades['損益(U)'].sum()
    total_unrealized = res_df[res_df['sort_group'] == 1]['損益(U)'].sum()
    final_equity = INITIAL_BALANCE + total_realized + total_unrealized
    total_roi = ((final_equity - INITIAL_BALANCE) / INITIAL_BALANCE) * 100

    # 3. 格式化輸出
    final_df = pd.DataFrame()
    final_df['時間'] = res_df['時間']
    final_df['幣種'] = res_df['幣種']
    final_df['方向'] = res_df['方向']
    final_df['狀態'] = res_df['狀態']
    final_df['進場價'] = res_df['進場價'].apply(lambda x: f"{x:,.4f}")
    final_df['現價/出場'] = res_df['現價/出場'].apply(lambda x: f"{x:,.4f}")
    final_df['報酬率%'] = res_df['報酬率%'].apply(lambda x: f"{x:+.2f}%")
    final_df['損益(U)'] = res_df['損益(U)'].apply(lambda x: f"{x:+.2f}")
    final_df['累計餘額'] = res_df['balance_calc'].apply(lambda x: f"{x:,.2f}")

    # 4. 輸出報表
    print(f"\n{'='*130}")
    print(f"資產績效報表 (本金: ${INITIAL_BALANCE:,.0f}) - {time.strftime('%Y-%m-%d %H:%M')}")
    print(f"{'='*130}")
    
    # 寬間距設定
    print(final_df.to_string(index=False, col_space=14, justify='right'))

    print("-" * 130)
    # 🔥 顯示統計區塊
    print(f"交易統計 (Stats)        : 總筆數 {total_closed} (勝 {win_count} / 負 {loss_count})")
    print(f"勝率 (Win Rate)         : {win_rate:.2f}%")
    print("-" * 130)
    print(f"已實現損益 (Realized)   : ${total_realized:,.2f}")
    print(f"浮動損益 (Floating)     : ${total_unrealized:,.2f}")
    print(f"當前總淨值 (Total Equity): ${final_equity:,.2f}")
    print(f"總報酬率 (Total ROI)    : {total_roi:+.2f}%")
    print(f"{'='*130}\n")

if __name__ == "__main__":
    check_performance()