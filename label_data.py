import pandas as pd
import numpy as np
from numba import njit
import os

# ==========================================
# 核心運算區 (由 Numba JIT 加速)
# ==========================================
@njit(parallel=False)
def calculate_labels_fast(close, high, low, atr, look_ahead, risk_reward, sl_multiplier):
    """
    使用 Numba 編譯加速的三重柵欄法標註邏輯
    """
    length = len(close)
    targets = np.zeros(length, dtype=np.int32) # 初始化標籤陣列 (全部為 0)
    
    # 遍歷每一根 K 線 (只跑到倒數第 N 根，因為最後幾根沒有未來可以看)
    for i in range(length - look_ahead):
        current_close = close[i]
        current_atr = atr[i]
        
        # 如果 ATR 無效，跳過
        if np.isnan(current_atr):
            continue
            
        # --- 設定該筆交易的計畫 ---
        # 做多 (Long)
        long_tp = current_close + (current_atr * sl_multiplier * risk_reward)
        long_sl = current_close - (current_atr * sl_multiplier)
        
        # 做空 (Short)
        short_tp = current_close - (current_atr * sl_multiplier * risk_reward)
        short_sl = current_close + (current_atr * sl_multiplier)
        
        long_signal = False
        short_signal = False
        
        # --- 往未來查看窗口 (Window) ---
        for j in range(1, look_ahead + 1):
            future_idx = i + j
            if future_idx >= length:
                break
            
            f_high = high[future_idx]
            f_low = low[future_idx]
            
            # 1. 檢查做多邏輯 (如果還沒確定失敗或成功)
            if not long_signal:
                # 嚴格判斷：如果這根 K 線最低價碰到止損，這單就廢了 (優先止損)
                if f_low <= long_sl:
                    long_signal = False # 失敗
                elif f_high >= long_tp:
                    # 如果沒碰到止損，且高點碰到止盈 -> 成功
                    long_signal = True
            
            # 2. 檢查做空邏輯
            if not short_signal:
                if f_high >= short_sl:
                    short_signal = False # 失敗 (碰到止損)
                elif f_low <= short_tp:
                    short_signal = True # 成功
            
            # 如果已經確定有方向成功，提早結束搜尋 (效率優化)
            if long_signal:
                targets[i] = 1 # Label 1: Buy
                break
            elif short_signal:
                targets[i] = 2 # Label 2: Sell
                break

    return targets

# ==========================================
# 資料處理區 (Pandas 介面)
# ==========================================
def apply_labels(df):
    """
    主呼叫函式：處理 DataFrame 並呼叫加速核心
    """
    print("[LABEL] 正在進行數據標註 (Numba 加速版)...")
    
    # --- 1. 參數設定 (可在此調整) ---
    RISK_REWARD_RATIO = 2.0  # 獲利目標: 2倍
    SL_ATR_MULTIPLIER = 1.0  # 止損距離: 1倍 ATR
    LOOK_AHEAD_CANDLES = 20  # 觀察窗口: 20根 K線
    
    # --- 2. 數據清洗與準備 ---
    # Numba 不喜歡吃 Pandas Series，要轉成 Numpy Array (float64)
    df = df.replace([np.inf, -np.inf], np.nan)
    
    # 提取需要的陣列
    close_arr = df['close'].values.astype(np.float64)
    high_arr = df['high'].values.astype(np.float64)
    low_arr = df['low'].values.astype(np.float64)
    atr_arr = df['ATR'].values.astype(np.float64)
    
    # --- 3. 呼叫加速運算 ---
    # 第一次執行會花幾秒鐘編譯，之後會瞬間完成
    targets = calculate_labels_fast(
        close_arr, 
        high_arr, 
        low_arr, 
        atr_arr, 
        LOOK_AHEAD_CANDLES, 
        RISK_REWARD_RATIO, 
        SL_ATR_MULTIPLIER
    )
    
    # --- 4. 將結果存回 DataFrame ---
    df['target'] = targets
    
    # 統計分佈
    counts = df['target'].value_counts()
    print(f"   [RESULT] 標註完成: {counts.to_dict()}")
    print(f"      (0=觀望/虧損, 1=做多獲利, 2=做空獲利)")
    
    # 移除最後一段沒有標註到的資料 (因為沒有未來數據)
    df_clean = df.iloc[:-LOOK_AHEAD_CANDLES].copy()
    
    return df_clean

# ==========================================
# 獨立測試區
# ==========================================
if __name__ == "__main__":
    # 測試用路徑
    input_file = 'data/ETHUSDT/4h_with_indicators.csv'
    output_file = 'data/ETHUSDT/4h_labeled.csv'
    
    if os.path.exists(input_file):
        try:
            df = pd.read_csv(input_file)
            
            # 執行標註
            df_labeled = apply_labels(df)
            
            # 存檔
            df_labeled.to_csv(output_file, index=False)
            print(f"[SUCCESS] 檔案已儲存至: {output_file}")
            
        except Exception as e:
            print(f"[ERROR] 發生錯誤: {e}")
    else:
        print(f"[WARNING] 找不到測試檔案: {input_file}，請先執行 add_indicators.py")