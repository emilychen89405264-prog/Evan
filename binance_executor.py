import ccxt
import time

# ==========================================
# 🔑 Binance 設定區
# 1. 如果只想跑模擬 (Paper Trading)，這裡留空即可
# 2. 如果要真實下單 (Real Trading)，請填入 Binance API Key
# ==========================================
API_KEY = '' 
SECRET_KEY = ''

def get_exchange():
    """
    初始化並連線到 Binance
    """
    exchange = ccxt.binance({
        'apiKey': API_KEY,
        'secret': SECRET_KEY,
        'enableRateLimit': True,
        'options': {
            'defaultType': 'future', 
        }
    })
    
    # ✅ 強制指定使用真實 URL (保險起見)
    exchange.urls['api'] = {
        'public': 'https://fapi.binance.com/fapi/v1',
        'private': 'https://fapi.binance.com/fapi/v1',
    }
    
    return exchange

def get_ticker_price(symbol):
    """
    查詢 Binance 最新成交價 (無需 API Key 即可查詢)
    """
    try:
        exchange = get_exchange()
        # 符號清洗：把 'BTC/USDT:USDT' 轉成 'BTC/USDT' 以符合 Binance 格式
        clean_symbol = symbol.split(':')[0] 
        
        ticker = exchange.fetch_ticker(clean_symbol)
        return float(ticker['last'])
    except Exception as e:
        # print(f"[Binance] 查價失敗 {symbol}: {e}") # 避免洗版，註解掉
        return None

def get_account_balance():
    """
    查詢 USDT 餘額
    - 如果有 API Key -> 查真實餘額
    - 如果無 API Key -> 回傳假餘額 $50,000 (模擬用)
    """
    if not API_KEY or not SECRET_KEY:
        # print("[模擬] 無 API Key，使用預設本金 $50,000")
        return 50000.0
    
    try:
        exchange = get_exchange()
        balance = exchange.fetch_balance()
        
        # Binance 合約帳戶的 USDT 通常在 total -> USDT
        if 'USDT' in balance['total']:
            return float(balance['total']['USDT'])
        else:
            return 0.0
    except Exception as e:
        print(f"[Binance] 查詢餘額失敗: {e}")
        return 0.0

def execute_real_trade(symbol, action, quantity_usdt, leverage=1, tp_price=None, sl_price=None):
    """
    發送訂單到 Binance
    - 如果無 Key -> 僅印出 Log 並回傳 True (讓外層 Paper Trader 記帳)
    - 如果有 Key -> 發送真實合約訂單
    """
    # 1. 檢查是否為模擬模式
    if not API_KEY or not SECRET_KEY:
        # print(f"[模擬執行] Binance 虛擬下單: {action} {symbol} ${quantity_usdt:.2f}")
        return True # 回傳 True 代表「執行成功」(其實是模擬成功)

    exchange = get_exchange()
    clean_symbol = symbol.split(':')[0] # 清洗符號

    try:
        # 2. 設定槓桿 (真實交易才需要)
        try:
            exchange.set_leverage(leverage, clean_symbol)
        except Exception as e:
            pass # 槓桿可能已經設過，忽略錯誤

        # 3. 取得價格並計算數量
        price = get_ticker_price(clean_symbol)
        if not price: return False
        
        amount = quantity_usdt / price
        
        # 精度處理 (Binance 對數量精度很敏感)
        amount = float(exchange.amount_to_precision(clean_symbol, amount))
        
        side = 'buy' if action == 'BUY' else 'sell'
        
        print(f"[Binance] 正在下單: {side} {amount} {clean_symbol} (x{leverage})...")

        # 4. 發送主訂單 (市價單)
        # 注意：Binance API 的 TP/SL 通常建議分開掛單，這裡為了穩定性，先發送主單
        order = exchange.create_order(
            symbol=clean_symbol,
            type='market',
            side=side,
            amount=amount
        )
        
        print(f"[成功] Binance 訂單成交！ID: {order['id']}")
        
        # 5. (進階) 如果成交成功，這裡可以補掛 TP/SL 單
        # 為了避免代碼過於複雜導致報錯，這裡暫時略過自動掛止盈止損單的邏輯
        # 建議真實交易時，配合交易所的「自動止盈止損」設定，或在程式碼中追加 OCO 訂單
        
        return True

    except Exception as e:
        print(f"[Binance] 下單失敗: {e}")
        return False

if __name__ == "__main__":
    # 測試用：印出目前的 BTC 價格
    price = get_ticker_price("BTC/USDT")
    print(f"目前 Binance BTC 價格: {price}")
    
    # 測試用：印出餘額
    bal = get_account_balance()
    print(f"目前帳戶餘額: ${bal}")