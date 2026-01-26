import pandas as pd
import numpy as np
import joblib  # 用來儲存 Scaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

def create_sequences(X, y, time_steps=60):
    Xs, ys = [], []
    for i in range(len(X) - time_steps):
        Xs.append(X[i:(i + time_steps)])
        ys.append(y[i + time_steps])
    return np.array(Xs), np.array(ys)

if __name__ == "__main__":
    # 1. 讀取資料
    input_file = 'data/ETHUSDT_4h_labeled.csv'
    print(f"讀取資料: {input_file}...")
    df = pd.read_csv(input_file)

    # 2. 特徵篩選
    feature_cols = ['RSI', 'Dist_EMA200', 'Dist_EMA50', 'Vol_Rel', 'ATR']
    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    
    X = df[feature_cols].values
    y = df['target'].values

    # 3. 數據正規化 (並儲存 Scaler!)
    # 這一步至關重要，我們把這個「量尺」存下來，以後預測都用這把尺
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    
    # 儲存 Scaler
    joblib.dump(scaler, 'models/scaler.pkl')
    print("Scaler 已儲存為 'scaler.pkl' (預測時必須用它)")

    # 4. 製作序列
    TIME_STEPS = 60
    X_seq, y_seq = create_sequences(X_scaled, y, TIME_STEPS)
    y_seq = to_categorical(y_seq, num_classes=3)

    # 5. 切分數據 (不打亂順序，因為是時間序列)
    X_train, X_test, y_train, y_test = train_test_split(X_seq, y_seq, test_size=0.2, shuffle=False)

    # 6. 優化後的模型架構
    # 我們加入 BatchNormalization (加速收斂) 和調整 Dropout (防止過擬合)
    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])),
        Dropout(0.3), # 增加 Dropout 到 30%
        BatchNormalization(), # 讓數據分佈更穩定

        LSTM(64, return_sequences=False),
        Dropout(0.3),
        BatchNormalization(),

        Dense(32, activation='relu'), # 多加一層全連接層來整合特徵
        Dropout(0.2),

        Dense(3, activation='softmax')
    ])

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # 7. 設定訓練策略 (Callbacks)
    callbacks = [
        # 早停機制: 如果 'val_loss' 連續 5 次沒有變好，就提早結束
        EarlyStopping(monitor='val_loss', patience=5, verbose=1, restore_best_weights=True),
        
        # 存檔機制: 只儲存 'val_accuracy' 最高的那一次模型
        ModelCheckpoint('models/best_crypto_model.keras', monitor='val_accuracy', save_best_only=True, verbose=1)
    ]

    # 8. 開始訓練
    print("開始優化訓練...")
    history = model.fit(
        X_train, y_train,
        epochs=50,          # 設定 50，但通常會被 EarlyStopping 提早擋下
        batch_size=32,
        validation_data=(X_test, y_test),
        callbacks=callbacks # 把策略掛上去
    )

    # 9. 最終評估 (這會顯示最佳模型的成績，而不是最後一次的成績)
    print("\n驗證集最終成績:")
    loss, accuracy = model.evaluate(X_test, y_test)
    print(f"Accuracy: {accuracy * 100:.2f}%")