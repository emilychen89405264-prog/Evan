import os
import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from fetch_data import fetch_historical_data
from add_indicators import add_technical_indicators
from label_data import apply_labels

# 設定路徑
DATA_DIR = 'data/'
MODELS_DIR = 'models/'

# 確保資料夾存在
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

def create_sequences(X, y, time_steps=60):
    Xs, ys = [], []
    for i in range(len(X) - time_steps):
        Xs.append(X[i:(i + time_steps)])
        ys.append(y[i + time_steps])
    return np.array(Xs), np.array(ys)

def run_pipeline_for_coin(symbol, timeframe='4h'):
    """
    一鍵完成：抓資料 -> 算指標 -> 標註 -> 訓練 -> 存檔
    """
    clean_symbol = symbol.replace('/', '')
    
    # ✅ 設定該幣種的專屬資料目錄: data/ETHUSDT/
    coin_data_dir = os.path.join(DATA_DIR, clean_symbol)
    os.makedirs(coin_data_dir, exist_ok=True) # 確保目錄存在

    print(f"\n[MANAGER] 正在啟動 {symbol} 的 AI 訓練流程...")

    # 1. 抓取資料 (fetch_data 已經改過，會自動存到 data/ETHUSDT/)
    print(f"   1. 下載歷史數據...")
    df = fetch_historical_data(symbol, timeframe, '2022-01-01 00:00:00')
    if len(df) < 300:
        print(f"[WARNING] {symbol} 歷史數據過短 (僅 {len(df)} 筆)，無法計算 EMA200，跳過訓練。")
        return False
    
    # 2. 計算指標
    print(f"   2. 計算技術指標...")
    df = add_technical_indicators(df)
    
    # 3. 數據標註
    print(f"   3. 進行策略標註 (Labeling)...")
    df = apply_labels(df)
    
    # ✅ 修改存檔路徑：存到小資料夾內 (data/ETHUSDT/4h_labeled.csv)
    labeled_filename = f"{timeframe}_labeled.csv"
    csv_path = os.path.join(coin_data_dir, labeled_filename)
    df.to_csv(csv_path, index=False)
    print(f"標註資料已備份至: {csv_path}")

    # 4. 準備訓練數據
    print(f"   4. 數據前處理與正規化...")
    feature_cols = ['RSI', 'Dist_EMA200', 'Dist_EMA50', 'Vol_Rel', 'ATR']
    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    
    X = df[feature_cols].values
    y = df['target'].values
    
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    
    # ✅ Scaler 建議也改名或分類，這裡我們先維持用檔名區分，但存到 models/
    scaler_path = os.path.join(MODELS_DIR, f"{clean_symbol}_scaler.pkl")
    joblib.dump(scaler, scaler_path)
    
    # 動態儲存 Scaler (重要！每個幣種要有自己的尺)
    scaler_path = os.path.join(MODELS_DIR, f"{clean_symbol}_scaler.pkl")
    joblib.dump(scaler, scaler_path)
    
    X_seq, y_seq = create_sequences(X_scaled, y, 60)
    y_seq = to_categorical(y_seq, num_classes=3)
    
    if len(X_seq) < 100:
        print(f"[ERROR] {symbol} 的有效數據過少，跳過訓練。")
        return False

    X_train, X_test, y_train, y_test = train_test_split(X_seq, y_seq, test_size=0.2, shuffle=False)

    # 5. 建立模型
    print(f"5. 訓練神經網絡中...")
    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])),
        Dropout(0.3),
        BatchNormalization(),
        LSTM(64, return_sequences=False),
        Dropout(0.3),
        BatchNormalization(),
        Dense(32, activation='relu'),
        Dropout(0.2),
        Dense(3, activation='softmax')
    ])
    
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    # 動態儲存模型路徑
    model_path = os.path.join(MODELS_DIR, f"{clean_symbol}_model.keras")
    
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
        ModelCheckpoint(model_path, monitor='val_accuracy', save_best_only=True, verbose=0)
    ]
    
    model.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_test, y_test), callbacks=callbacks, verbose=0)
    
    print(f"[SUCCESS] {symbol} 模型訓練完成！已儲存至 {model_path}")
    return True