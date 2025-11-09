# Symbol Format Guide

## How to Use Custom Symbols

The application now supports custom symbol input! Select "🔍 Other/Custom Symbol" from the dropdown and enter any valid Yahoo Finance ticker.

## Symbol Formats

### Stocks (US Markets)

**Format:** `TICKER` (uppercase letters)

**Examples:**
```
TSLA    - Tesla
NFLX    - Netflix
AMD     - Advanced Micro Devices
NVDA    - NVIDIA
DIS     - Disney
BA      - Boeing
JPM     - JPMorgan Chase
V       - Visa
WMT     - Walmart
KO      - Coca-Cola
```

**How to find:**
1. Go to Yahoo Finance
2. Search for the company
3. The ticker is shown next to the company name

### Cryptocurrency

**Format:** `TICKER-USD` (crypto ticker + "-USD")

**Examples:**
```
BTC-USD     - Bitcoin
ETH-USD     - Ethereum
BNB-USD     - Binance Coin
ADA-USD     - Cardano
SOL-USD     - Solana
DOGE-USD    - Dogecoin
XRP-USD     - Ripple
DOT-USD     - Polkadot
MATIC-USD   - Polygon
AVAX-USD    - Avalanche
```

**Other crypto pairs:**
```
BTC-EUR     - Bitcoin in Euros
ETH-GBP     - Ethereum in British Pounds
```

### Foreign Exchange (ForEx)

**Format:** `XXXYYY=X` (currency pair + "=X")

**Examples:**
```
EURUSD=X    - Euro to US Dollar
GBPUSD=X    - British Pound to US Dollar
JPYUSD=X    - Japanese Yen to US Dollar
AUDUSD=X    - Australian Dollar to US Dollar
CADUSD=X    - Canadian Dollar to US Dollar
CHFUSD=X    - Swiss Franc to US Dollar
NZDUSD=X    - New Zealand Dollar to US Dollar
```

### International Stocks

**Format varies by exchange:**

**London Stock Exchange (UK):**
```
BP.L        - BP
HSBA.L      - HSBC
VOD.L       - Vodafone
```

**Toronto Stock Exchange (Canada):**
```
SHOP.TO     - Shopify
RY.TO       - Royal Bank of Canada
```

**Tokyo Stock Exchange (Japan):**
```
7203.T      - Toyota
9984.T      - SoftBank
```

**Hong Kong Stock Exchange:**
```
0700.HK     - Tencent
9988.HK     - Alibaba
```

### ETFs (Exchange-Traded Funds)

**Format:** `TICKER` (same as stocks)

**Examples:**
```
SPY         - S&P 500 ETF
QQQ         - Nasdaq 100 ETF
DIA         - Dow Jones ETF
IWM         - Russell 2000 ETF
VTI         - Total Stock Market ETF
VOO         - Vanguard S&P 500 ETF
```

### Indices

**Format:** `^TICKER` (caret + ticker)

**Examples:**
```
^GSPC       - S&P 500 Index
^DJI        - Dow Jones Industrial Average
^IXIC       - NASDAQ Composite
^RUT        - Russell 2000
^VIX        - Volatility Index
^FTSE       - FTSE 100 (UK)
^N225       - Nikkei 225 (Japan)
```

## Common Symbols Quick Reference

### Tech Giants
```
AAPL    - Apple
GOOGL   - Google (Alphabet Class A)
GOOG    - Google (Alphabet Class C)
MSFT    - Microsoft
AMZN    - Amazon
META    - Meta (Facebook)
NVDA    - NVIDIA
TSLA    - Tesla
```

### Popular Stocks
```
NFLX    - Netflix
DIS     - Disney
BA      - Boeing
NKE     - Nike
SBUX    - Starbucks
MCD     - McDonald's
WMT     - Walmart
TGT     - Target
```

### Top Cryptocurrencies
```
BTC-USD     - Bitcoin
ETH-USD     - Ethereum
BNB-USD     - Binance Coin
XRP-USD     - Ripple
ADA-USD     - Cardano
SOL-USD     - Solana
DOGE-USD    - Dogecoin
DOT-USD     - Polkadot
```

### Major ForEx Pairs
```
EURUSD=X    - EUR/USD
GBPUSD=X    - GBP/USD
USDJPY=X    - USD/JPY
AUDUSD=X    - AUD/USD
USDCAD=X    - USD/CAD
```

## Tips for Finding Symbols

### Method 1: Yahoo Finance
1. Go to https://finance.yahoo.com
2. Search for the asset
3. The symbol is displayed prominently
4. Copy and paste into the app

### Method 2: Google Search
```
Search: "Tesla stock symbol"
Result: TSLA

Search: "Bitcoin ticker Yahoo Finance"
Result: BTC-USD
```

### Method 3: Company Website
- Most public companies list their ticker symbol
- Usually found in the "Investor Relations" section

## Troubleshooting

### Symbol Not Found

**Error:** "Failed to fetch data" or "Insufficient data"

**Solutions:**
1. **Check spelling** - Symbols are case-insensitive but must be correct
2. **Verify format** - Crypto needs "-USD", ForEx needs "=X"
3. **Try Yahoo Finance** - Search there first to confirm the symbol
4. **Check if delisted** - Some companies may no longer trade

**Examples of common mistakes:**
```
❌ TESLA      → ✅ TSLA
❌ BTC        → ✅ BTC-USD
❌ EURUSD     → ✅ EURUSD=X
❌ apple      → ✅ AAPL (case doesn't matter, but symbol must be correct)
```

### Low Data Quality

**Issue:** High error rates or poor predictions

**Possible causes:**
1. **Penny stocks** - Very low volume, hard to predict
2. **New listings** - Not enough historical data
3. **Delisted stocks** - No longer trading
4. **Exotic pairs** - Limited data availability

**Recommendation:**
- Stick to major stocks, popular cryptos, and main ForEx pairs
- These have better data quality and more predictable patterns

## Symbol Validation

The app will attempt to fetch data for any symbol you enter. If the symbol is invalid or has insufficient data, you'll see an error message.

**Valid symbols will:**
- ✅ Fetch historical data successfully
- ✅ Display candlestick chart
- ✅ Generate predictions

**Invalid symbols will:**
- ❌ Show "Failed to fetch data" error
- ❌ Show "Insufficient data" error
- ❌ Not display any charts

## Advanced Usage

### Comparing Similar Assets

```
Compare:
GOOGL vs GOOG    - Different share classes
BTC-USD vs BTC-EUR - Same asset, different currency
SPY vs VOO       - Similar ETFs
```

### Sector Analysis

```
Tech:     AAPL, MSFT, GOOGL, NVDA
Finance:  JPM, BAC, GS, WFC
Energy:   XOM, CVX, COP, SLB
Retail:   WMT, TGT, COST, HD
```

### Portfolio Tracking

Use custom symbols to track your entire portfolio:
1. Enter each holding's symbol
2. Generate forecasts
3. Compare predictions across assets
4. Make informed decisions

## Need Help?

If you're unsure about a symbol:
1. Search on Yahoo Finance first
2. Verify the symbol format
3. Try the symbol in the app
4. Check the error message if it fails

Most popular stocks, cryptos, and ForEx pairs will work perfectly!
