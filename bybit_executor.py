import ccxt
import time

# --- 改用 Bybit Testnet ---
API_KEY = ''
SECRET_KEY = ''

def get_exchange():
    """
    初始化並連線到 Bybit Testnet
    """
    exchange = ccxt.bybit({
        'apiKey': API_KEY,
        'secret': SECRET_KEY,
        'enableRateLimit': True,
        'options': {
            'defaultType': 'future',  # 設定預設為合約交易
            # 'defaultType': 'linear', # 或是 linear (USDT本位)
        }
    })
    
    # ✅ 開啟測試模式 (Bybit 的 ccxt 支援這個指令，比幣安穩)
    exchange.set_sandbox_mode(True) 
    
    return exchange

def execute_real_trade(symbol, action, quantity_usdt, leverage=1, tp_price=None, sl_price=None):
    """
    發送真實訂單到 Bybit 測試網
    """
    exchange = get_exchange()
    
    try:
        # 1. 設定槓桿 (Bybit 需要指定 margin mode，這裡簡化處理)
        try:
            # Bybit 設定槓桿通常需要指定是 Cross(全倉) 還是 Isolated(逐倉)
            # 這裡嘗試設定為 逐倉 (Isolated) + 槓桿倍數
            exchange.set_leverage(leverage, symbol)
        except Exception as e:
            # 很多時候槓桿已經設過，重複設會報錯，忽略即可
            pass

        # 2. 取得最新價格
        ticker = exchange.fetch_ticker(symbol)
        current_price = ticker['last']
        
        # 3. 計算下單數量 (顆數)
        amount = (quantity_usdt * leverage) / current_price
        
        # Bybit 對數量精度要求很嚴格，這裡做簡單處理
        # 為了保險，Bybit 合約通常最小單位比較大，我們取小數點後3位試試
        # 正規做法是用 exchange.amount_to_precision(symbol, amount)
        amount = float(exchange.amount_to_precision(symbol, amount))

        side = 'buy' if action == 'BUY' else 'sell'
        
        print(f"[交易所] Bybit 正在下單: {side} {amount} {symbol}...")
        
        # 4. 準備參數 (止盈止損)
        params = {}
        if tp_price:
            params['takeProfit'] = str(tp_price)
        if sl_price:
            params['stopLoss'] = str(sl_price)
            
        # Bybit 支援在開倉時直接帶入 TP/SL，這很方便！
        order = exchange.create_order(
            symbol=symbol,
            type='market',
            side=side,
            amount=amount,
            params=params
        )
        
        print(f"[成功] Bybit 訂單成交！ID: {order['id']}")
        return True

    except Exception as e:
        print(f"[錯誤] 下單失敗: {e}")
        return False
    
def get_ticker_price(symbol):
    """
    查詢 Bybit 測試網的最新成交價
    """
    try:
        exchange = get_exchange()
        ticker = exchange.fetch_ticker(symbol)
        return float(ticker['last'])
    except Exception as e:
        print(f"[錯誤] 無法查詢 {symbol} 最新價格: {e}")
        return None

def get_account_balance():
    """
    查詢 Bybit 測試網 USDT 餘額
    """
    try:
        exchange = get_exchange()
        balance = exchange.fetch_balance()
        
        # Bybit 統一帳戶的 USDT 通常在 'total' 或 'free' 裡
        if 'USDT' in balance:
            usdt_balance = balance['USDT']['free']
        else:
            # 有時候在 unified account 結構不同，這裡做個備用
            usdt_balance = balance.get('total', {}).get('USDT', 0)
            
        print(f"[交易所] Bybit 帳戶可用餘額: ${usdt_balance:,.2f}")
        return usdt_balance
    except Exception as e:
        print(f"[錯誤] 查詢餘額失敗: {e}")
        return 0.0

if __name__ == "__main__":
    get_account_balance()