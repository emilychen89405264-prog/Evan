import pandas as pd
import pandas_ta as ta  # 引入技術分析庫

def add_technical_indicators(df):
    """
    為數據表加入技術指標
    """
    print("正在計算技術指標...")

    # 1. EMA (指數移動平均線) - 判斷趨勢
    # EMA 50: 中期趨勢
    # EMA 200: 長期趨勢 (牛熊分界線)
    df['EMA_50'] = df.ta.ema(length=50)
    df['EMA_200'] = df.ta.ema(length=200)

    # 2. RSI (相對強弱指標) - 判斷過熱或過冷
    df['RSI'] = df.ta.rsi(length=14)

    # 3. ATR (平均真實波幅) - 判斷波動率
    # 如果 ATR 很大，代表現在波動劇烈，停損要設遠一點
    # 如果 ATR 很小，代表現在是盤整，停損可以設近一點
    df['ATR'] = df.ta.atr(length=14)

    # --- 為了讓 AI 更容易學習，我們做一些數據正規化 (Normalization) ---
    
    # 特徵 A: 價格距離 EMA 200 有多遠？ (百分比)
    # 正數代表在趨勢線上，負數代表在趨勢線下
    df['Dist_EMA200'] = (df['close'] - df['EMA_200']) / df['EMA_200']

    # 特徵 B: 價格距離 EMA 50 有多遠？
    df['Dist_EMA50'] = (df['close'] - df['EMA_50']) / df['EMA_50']
    
    # 特徵 C: 成交量變化 (當前成交量 / 過去20根平均成交量)
    # 大於 1 代表爆量，小於 1 代表縮量
    df['Vol_Rel'] = df['volume'] / df['volume'].rolling(window=20).mean()

    # 清除因為計算指標產生的空值 (例如 EMA200 前 200 筆會是空的)
    df.dropna(inplace=True)
    
    print("指標計算完成！")
    return df

# --- 主程式 ---
if __name__ == "__main__":
    # 1. 讀取你剛才抓下來的檔案 (請確認檔名是否正確)
    input_file = 'ETHUSDT_4h_data.csv' 
    output_file = 'ETHUSDT_4h_with_indicators.csv'
    
    try:
        print(f"讀取檔案: {input_file}")
        df = pd.read_csv(input_file)
        
        # 確保數據夠多
        if len(df) < 200:
            print("數據量太少, 無法計算 EMA200, 請重新抓取更多數據。")
        else:
            # 2. 計算指標
            df_processed = add_technical_indicators(df)
            
            # 3. 檢查一下前 5 筆
            print("\n--- 預覽處理後的數據 ---")
            print(df_processed[['datetime', 'close', 'RSI', 'Dist_EMA200', 'ATR']].head())
            
            # 4. 存檔
            df_processed.to_csv(output_file, index=False)
            print(f"\n處理後的檔案已儲存為: {output_file}")
            
    except FileNotFoundError:
        print(f"找不到檔案: {input_file}, 請確認你是否已經執行過 fetch_data.py")