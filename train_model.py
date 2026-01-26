import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.utils import to_categorical

def create_sequences(X, y, time_steps=60):
    """
    將數據轉換為 LSTM 需要的 3D 格式 (Samples, Time Steps, Features)
    這就像是把一張張 K 線圖切出來給 AI 看
    """
    Xs, ys = [], []
    for i in range(len(X) - time_steps):
        Xs.append(X[i:(i + time_steps)])
        ys.append(y[i + time_steps])
    return np.array(Xs), np.array(ys)

# --- 主程式 ---
if __name__ == "__main__":
    # 1. 讀取標註好的資料
    input_file = 'ETHUSDT_4h_labeled.csv'
    print(f"讀取資料: {input_file}...")
    df = pd.read_csv(input_file)

    # 2. 特徵篩選 (Feature Selection)
    # 我們只選取「AI 能夠理解的指標」，不選原始價格 (Close/High/Low)
    # 這樣 AI 才能學會「型態」而不是死背「價格」
    feature_cols = ['RSI', 'Dist_EMA200', 'Dist_EMA50', 'Vol_Rel', 'ATR']
    
    print(f"使用特徵: {feature_cols}")
    
    # 檢查是否有無限大 (inf) 或空值 (nan) 的髒資料，並清除
    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    
    X = df[feature_cols].values
    y = df['target'].values

    # 3. 數據正規化 (Normalization)
    # 把所有特徵縮放到 0 到 1 之間，這是神經網絡訓練的必要步驟
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    # 4. 製作時間序列數據 (Sequencing)
    # 設定 AI 要回看多少根 K 線？這裡設 60 根 (約 10 天的 4h 線)
    TIME_STEPS = 60
    print(f"正在製作時間序列數據 (回看 {TIME_STEPS} 根 K 線)...")
    X_seq, y_seq = create_sequences(X_scaled, y, TIME_STEPS)
    
    # 將標籤轉為 One-hot encoding (例如 2 變成 [0, 0, 1])
    y_seq = to_categorical(y_seq, num_classes=3)

    # 5. 切分訓練集與測試集
    # 80% 用來訓練，20% 用來考試
    X_train, X_test, y_train, y_test = train_test_split(X_seq, y_seq, test_size=0.2, shuffle=False)
    
    print(f"訓練集大小: {X_train.shape}")
    print(f"測試集大小: {X_test.shape}")

    # 6. 搭建 LSTM 模型 (大腦結構)
    print("正在建立 LSTM 模型...")
    model = Sequential([
        # 第一層 LSTM
        LSTM(units=64, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])),
        Dropout(0.2), # 丟棄 20% 神經元，防止死記硬背 (Overfitting)
        
        # 第二層 LSTM
        LSTM(units=64, return_sequences=False),
        Dropout(0.2),
        
        # 輸出層 (3 個神經元，分別代表 0, 1, 2 的機率)
        Dense(units=3, activation='softmax')
    ])

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # 7. 開始訓練 (Training)
    print("開始訓練 AI (這可能需要幾分鐘)...")
    history = model.fit(
        X_train, y_train,
        epochs=20,          # 總共讀書讀 20 遍
        batch_size=32,      # 每次讀 32 題
        validation_data=(X_test, y_test),
        verbose=1
    )

    # 8. 儲存模型
    model.save('crypto_trader_model.h5')
    print("\n模型已儲存為: crypto_trader_model.h5")
    
    # 9. 簡單評估
    loss, accuracy = model.evaluate(X_test, y_test)
    print(f"\n最終測試準確率: {accuracy * 100:.2f}%")