# AI Crypto Portfolio Manager (AI 加密貨幣基金經理人)

An intelligent, automated cryptocurrency trading system with adaptive strategies (Trend/Leverage) and paper trading capabilities.

**Languages / 語言:**
[🇺🇸 English](#-english-version) | [🇹🇼 繁體中文](#-traditional-chinese-version)

---

<a name="-english-version"></a>
## 🇺🇸 English Version

###  Introduction
**AI Crypto Portfolio Manager** is a fully automated trading bot designed for local execution. It autonomously scans the Binance market for high-volume assets, trains custom LSTM deep learning models for each coin, and executes paper trades based on an adaptive strategy engine.

It features a **"Fund Manager"** architecture that dynamically manages a portfolio of coins, deciding when to enter/exit and how much leverage to use based on market volatility.

### Key Features

* **Smart Market Scanning:** Automatically scans Binance for top-volume cryptocurrencies and filters out low-volatility "dead" coins.
* **Auto-Training Pipeline:** Handles the entire lifecycle: fetching data, calculating indicators, labeling strategies, and training LSTM models without human intervention.
* **Adaptive Strategy Engine:**
    * **Trend Analysis (ADX):** Automatically detects market regime. Uses **Trend Following** strategies for strong trends (ADX > 25) and **Mean Reversion** for ranging markets.
    * **Dynamic Leverage (ATR%):** Adjusts leverage (1x - 20x) based on asset volatility. High volatility = Low leverage; Low volatility = High leverage.
* **Paper Trading System:** A built-in ledger system that records trades, strategies used, and monitors Take Profit (TP) / Stop Loss (SL) in real-time.
* **Visual Performance Report:** Generates professional financial reports and Mark-to-Market equity curves including floating PnL.

### Project Structure

| File | Role | Description |
| :--- | :--- | :--- |
| `fund_manager.py` | **Main Controller** | The "Boss". Scans markets, manages model training, and triggers trade signals. |
| `paper_trader.py` | **Accountant** | Manages the `paper_portfolio.csv` ledger, records trades (with leverage/strategy), and monitors PnL. |
| `check_performance.py`| **Analyst** | Generates detailed PnL tables and visualizes the equity curve (Balance + Floating PnL). |
| `trainer_interface.py`| **Trainer** | Handles the ETL pipeline: Fetch Data -> Indicators -> Labeling -> Model Training. |
| `predict_optimized.py`| **Worker** | Processes data for inference and calculates strategy logic (ADX & Leverage suggestions). |
| `fetch_data.py` | Utility | Downloads historical OHLCV data from Binance. |
| `add_indicators.py` | Utility | Calculates technical indicators (RSI, EMA, ATR, ADX, etc.). |
| `label_data.py` | Utility | Labels data for training using the Triple Barrier Method. |
| `data/` | Storage | Stores historical data, CSV logs, and performance charts. |
| `models/` | Storage | Stores trained `.keras` AI models and `.pkl` scalers. |

### Installation

#### 1. Prerequisites
* Python 3.10 or higher.
* A stable internet connection (to fetch Binance public data).

#### 2. Install Dependencies
Run the following command in your terminal:

```bash
pip install pandas numpy ccxt tensorflow joblib pandas_ta scikit-learn schedule matplotlib
```
### How to Use
#### 1. Start the Fund Manager
This is the main program. It will run continuously, scanning the market every 4 hours.
Run the following command in your terminal:
```bash
python fund_manager.py
```
* Initial Run: The system will download data and train models for the top 10 coins. This may take 10-20 minutes. Please be patient.

* Routine: Once models are ready, it will predict and execute paper trades automatically.

* Note: Keep the terminal window open.

#### 2. View Performance
You can generate a report at any time without stopping the bot.

```Bash
python check_performance.py
```
Output:

* A detailed table in the terminal showing open/closed positions, strategies used, and leverage.

* A performance chart saved at `data/performance_chart.png`.

#### Configuration
You can adjust settings in `fund_manager.py`:

* `INITIAL_CAPITA`L: Simulation starting balance (Default: $50,000).

* `TARGET_ACTIVE_COINS`: Number of coins to maintain in the active pool (Default: 10).

* `SCAN_POOL_SIZE`: Number of candidates to scan from the market (Default: 50).


<a name="-traditional-chinese-version"></a>
## 中文版本

### 簡介
**AI Crypto Portfolio Manager (AI 加密貨幣基金經理人)** 是一個專為本地端執行設計的全自動交易機器人。它自主掃描 Binance 市場尋找高成交量資產，為每種貨幣訓練客製化的 LSTM 深度學習模型，並根據適應性策略引擎執行模擬交易。

它採用 **「基金經理人」** 架構，動態管理貨幣投資組合，根據市場波動性決定進出場時機以及使用多少槓桿。

### 核心功能

* **智慧市場掃描：** 自動掃描 Binance 上成交量最高的加密貨幣，並過濾掉低波動性的「死魚」幣。
* **自動訓練流程：** 處理整個生命週期：無需人工干預即可獲取數據、計算指標、標註策略和訓練 LSTM 模型。
* **適應性策略引擎：**
    * **趨勢分析 (ADX)：** 自動偵測市場狀態。針對強趨勢 (ADX > 25) 使用 **趨勢跟隨** 策略，針對盤整市場使用 **均值回歸** 策略。
    * **動態槓桿 (ATR%)：** 根據資產波動性調整槓桿 (1x - 20x)。高波動 = 低槓桿；低波動 = 高槓桿。
* **模擬交易系統：** 內建的帳本系統，記錄交易、使用的策略，並即時監控止盈 (TP) / 止損 (SL)。
* **視覺化績效報告：** 生成專業的財務報告和包含浮動損益的市值計價 (Mark-to-Market) 資產曲線。

### 專案結構

| 檔案 | 角色 | 描述 |
| :--- | :--- | :--- |
| `fund_manager.py` | **總控制器** | 「老闆」。掃描市場、管理模型訓練並觸發交易信號。 |
| `paper_trader.py` | **會計師** | 管理 `paper_portfolio.csv` 帳本，記錄交易（含槓桿/策略），並監控損益。 |
| `check_performance.py`| **分析師** | 生成詳細的損益表並視覺化資產曲線（餘額 + 浮動損益）。 |
| `trainer_interface.py`| **訓練師** | 處理 ETL 流程：獲取數據 -> 指標 -> 標註 -> 模型訓練。 |
| `predict_optimized.py`| **工人** | 處理推理數據並計算策略邏輯（ADX 和槓桿建議）。 |
| `fetch_data.py` | 工具 | 從 Binance 下載歷史 OHLCV 數據。 |
| `add_indicators.py` | 工具 | 計算技術指標（RSI, EMA, ATR, ADX 等）。 |
| `label_data.py` | 工具 | 使用三重柵欄法標註訓練數據。 |
| `data/` | 儲存 | 儲存歷史數據、CSV 日誌和績效圖表。 |
| `models/` | 儲存 | 儲存訓練好的 `.keras` AI 模型和 `.pkl` 縮放器。 |

### 安裝

#### 1. 前置需求
* Python 3.10 或更高版本。
* 穩定的網路連線（用於獲取 Binance 公開數據）。

#### 2. 安裝依賴項
在你的終端機中執行以下指令：

```bash
pip install pandas numpy ccxt tensorflow joblib pandas_ta scikit-learn schedule matplotlib
```
### 如何使用
#### 1. 啟動基金經理人
這是主程式。它將持續運行，每 4 小時掃描一次市場。 在你的終端機中執行以下指令：
```bash
python fund_manager.py
```
* 初次運行： 系統將下載數據並訓練前 10 大貨幣的模型。這可能需要 10-20 分鐘。請耐心等待。

* 例行運行： 一旦模型準備就緒，它將自動預測並執行模擬交易。

* 注意： 請保持終端機視窗開啟。

#### 2. 查看績效
你可以在不停止機器人的情況下隨時生成報告。

```Bash
python check_performance.py
```
輸出：

* 終端機中顯示詳細表格，包含開倉/平倉部位、使用的策略和槓桿。

* 儲存在 `data/performance_chart.png` 的績效圖表。

#### 設定
你可以在 `fund_manager.py` 中調整設定：

* `INITIAL_CAPITAL`: 模擬起始餘額 (預設: $50,000)。

* `TARGET_ACTIVE_COINS`: 維持在活躍池中的貨幣數量 (預設: 10)。

* `SCAN_POOL_SIZE`: 從市場掃描的候選數量 (預設: 50)。


