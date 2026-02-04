import ccxt
import time

# ==========================================
# 🔑 Bybit 設定區
# 請填入從 https://testnet.bybit.com/user/assets/home/settings/api-management 取得的 Key
# ==========================================
API_KEY = ''
SECRET_KEY = ''

def check_account_status():
    print(f"[{time.strftime('%H:%M:%S')}] 正在連線到 Bybit Testnet (合約)...")
    print("------------------------------------------------------")
    
    try:
        # 初始化交易所
        exchange = ccxt.bybit({
            'apiKey': API_KEY,
            'secret': SECRET_KEY,
            'enableRateLimit': True,
            'options': {
                'defaultType': 'linear',  # linear = USDT本位合約 (最常用)
            }
        })
        
        # ✅ 開啟測試模式 (這行最重要，會自動連去 testnet.bybit.com)
        exchange.set_sandbox_mode(True)
        
        # 1. 查詢餘額
        # Bybit 的餘額結構比較複雜 (Unified Account vs Standard)
        # 我們嘗試抓取 USDT 的總餘額
        balance = exchange.fetch_balance()
        
        print("連線成功！帳戶資金如下：")
        
        usdt_balance = 0
        if 'USDT' in balance:
            usdt_balance = balance['USDT']['free']
            print(f"USDT 可用餘額: ${usdt_balance:,.2f}")
            print(f"USDT 總權益  : ${balance['USDT']['total']:,.2f}")
        else:
            # 備用查詢：有時候在 unified 結構裡
            usdt_balance = balance.get('total', {}).get('USDT', 0)
            print(f"USDT 餘額: ${usdt_balance:,.2f}")

        if usdt_balance < 10:
            print(" 餘額似乎有點少，記得去 Bybit 測試網領水龍頭資金！")

# 2. 檢查當前持倉 (修正版)
        print("\n檢查當前持倉 (Positions)...")
        try:
            # 🔴 [修正] 不傳入參數，直接抓取「所有」持倉，再由程式過濾
            all_positions = exchange.fetch_positions()
            
            has_position = False
            for p in all_positions:
                size = float(p['contracts']) # 持倉數量
                
                # 只顯示有倉位 (數量 > 0) 的幣種
                if size > 0:
                    has_position = True
                    symbol = p['symbol']
                    side = p['side'].upper() # long / short
                    leverage = p['leverage']
                    entry_price = float(p['entryPrice'])
                    pnl = float(p['unrealizedPnl'] or 0)
                    
                    print(f"    {symbol} [{side}] x{leverage}")
                    print(f"      數量: {size} | 進場: {entry_price} | 未實現損益: {pnl:.4f} U")
            
            if not has_position:
                print("   目前無任何持倉。")
                
        except Exception as e:
            print(f"   (持倉查詢失敗: {e})")

        print("------------------------------------------------------")
        print(" Bybit 狀態檢查完成！")

    except Exception as e:
        print(f"[錯誤] 發生錯誤: {e}")

if __name__ == "__main__":
    check_account_status()