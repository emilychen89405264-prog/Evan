import pandas as pd
import numpy as np
from tqdm import tqdm  # 進度條套件

def apply_labels(df):
    """
    使用 ATR 進行動態止盈止損標註
    獲利因子目標: 2.0 (賺 2 賠 1)
    """
    print("正在進行數據標註 (這可能需要幾分鐘)...")
    
    # 參數設定
    RISK_REWARD_RATIO = 2.0  # 賺 2
    SL_ATR_MULTIPLIER = 1.0  # 賠 1
    LOOK_AHEAD_CANDLES = 20  # 最多看未來 20 根 K 線 (約 3-4 天)
    
    # 初始化標籤欄位 (預設為 0: 不動作)
    df['target'] = 0 
    
    # 我們需要 Reset Index 以便用索引遍歷
    df = df.reset_index(drop=True)
    
    # 使用 tqdm 顯示進度條
    for i in tqdm(range(len(df) - LOOK_AHEAD_CANDLES)):
        
        # 1. 取得當前的進場條件
        current_close = df.loc[i, 'close']
        current_atr = df.loc[i, 'ATR']
        
        # 如果 ATR 是 NaN (前幾筆)，跳過
        if np.isnan(current_atr):
            continue
            
        # 2. 設定這筆交易的 止盈(TP) 與 止損(SL)
        # 做多 (Long) 的劇本
        long_tp = current_close + (current_atr * SL_ATR_MULTIPLIER * RISK_REWARD_RATIO)
        long_sl = current_close - (current_atr * SL_ATR_MULTIPLIER)
        
        # 做空 (Short) 的劇本
        short_tp = current_close - (current_atr * SL_ATR_MULTIPLIER * RISK_REWARD_RATIO)
        short_sl = current_close + (current_atr * SL_ATR_MULTIPLIER)
        
        # 3. 往未來查看 LOOK_AHEAD_CANDLES 根 K 線
        # 取得未來一段時間的高低點視窗
        future_window = df.loc[i+1 : i+LOOK_AHEAD_CANDLES]
        
        # --- 判斷做多 (Label 1) ---
        # 找出是否有任何一根K線的「最高價」超過 TP
        tp_hit = False
        sl_hit = False
        
        # 這裡我們簡化邏輯：檢查視窗內「最高價」是否觸及 TP，且「最低價」是否沒觸及 SL
        # 嚴謹的寫法應該要看「誰先發生」，但為了效率我們先檢查極值
        
        # 檢查哪一根 K 線先碰到 TP 或 SL
        for j in range(len(future_window)):
            future_idx = future_window.index[j]
            future_high = df.loc[future_idx, 'high']
            future_low = df.loc[future_idx, 'low']
            
            # 檢查做多邏輯
            if future_low <= long_sl:
                sl_hit = True # 先碰到停損，這單失敗
                break
            if future_high >= long_tp:
                tp_hit = True # 先碰到止盈，這單成功！
                break
        
        if tp_hit:
            df.loc[i, 'target'] = 1  # 標記為「適合做多」
            continue # 如果適合做多，就不用檢查做空了 (互斥)

        # --- 判斷做空 (Label 2) ---
        # 只有在不是做多的情況下才檢查做空
        tp_hit_short = False
        sl_hit_short = False
        
        for j in range(len(future_window)):
            future_idx = future_window.index[j]
            future_high = df.loc[future_idx, 'high']
            future_low = df.loc[future_idx, 'low']
            
            if future_high >= short_sl:
                sl_hit_short = True # 做空被嘎，失敗
                break
            if future_low <= short_tp:
                tp_hit_short = True # 做空獲利，成功
                break
                
        if tp_hit_short:
            df.loc[i, 'target'] = 2 # 標記為「適合做空」

    return df

# --- 主程式 ---
if __name__ == "__main__":
    # 確保你有安裝 tqdm (進度條)
    # pip install tqdm
    
    input_file = 'ETHUSDT_4h_with_indicators.csv'
    output_file = 'ETHUSDT_4h_labeled.csv'
    
    try:
        df = pd.read_csv(input_file)
        
        # 執行標註
        df_labeled = apply_labels(df)
        
        # 移除最後那 20 筆無法標註的資料 (因為沒有未來)
        df_labeled = df_labeled[:-20]
        
        # 檢查標註結果的分佈
        print("\n--- 標籤分佈狀況 ---")
        print(df_labeled['target'].value_counts())
        print("0 = 觀望/虧損, 1 = 做多獲利, 2 = 做空獲利")
        
        # 存檔
        df_labeled.to_csv(output_file, index=False)
        print(f"\n標註完成, 檔案已儲存為: {output_file}")
        
    except FileNotFoundError:
        print(f"找不到檔案: {input_file}")
    except ImportError:
        print("請先安裝 tqdm 套件: pip install tqdm")