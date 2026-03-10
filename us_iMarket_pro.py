import streamlit as st
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import mplfinance as mpf
import feedparser
import urllib.parse
from datetime import datetime
import google.generativeai as genai # 添加这一行

# --- 1. Basic Configuration ---
st.set_page_config(
    page_title="iMarket Professional | AI-Quant Terminal", 
    page_icon="⚖️", # 或者 "📈"
    layout="wide"
    )
# 修改前: def run_gemini_pro_analysis(ticker, tech_metrics, news_summary):
# 修改后:
def run_gemini_pro_analysis(ticker, tech_metrics, news_summary, language="中文"):
    if "GEMINI_API_KEY" in st.secrets:
        api_key_val = st.secrets["GEMINI_API_KEY"]
    else:
        return "❌ 错误：未在 Streamlit Cloud 后台配置 GEMINI_API_KEY"
    
    genai.configure(api_key=api_key_val)
    
    try:
        available_models = [m.name for m in genai.list_models() if 'generateContent' in m.supported_generation_methods]
        if not available_models:
            return "❌ 你的 API Key 没有任何可用模型。"
        
        target_model = next((m for m in available_models if 'flash' in m.lower()), available_models[0])
        model = genai.GenerativeModel(target_model)

        # --- 关键修改：根据选择切换 AI 的写作指令 ---
        role_instruction = "Top Wall Street Quant Strategist" if language == "English" else "顶级华尔街量化策略师"
        task_instruction = "write a deep research report" if language == "English" else "撰写深度研究报告"

        prompt = f"""
        As a {role_instruction}, please {task_instruction} for {ticker}.
        【Technical Metrics】: {tech_metrics}
        【Recent News】: {news_summary}
        Framework: Core Thesis, Technical Scan, Risk Analysis, and Investment Guidance.
        Output Language: {language}
        """

        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"❌ AI 分析出错: {str(e)}"
    
    
# --- 2. Top Market Indices (Color-Coded) ---
@st.cache_data(ttl=300)
def fetch_market_indices():
    indices = {
        "DJIA": "^DJI", "NDX": "^NDX", "SPX": "^GSPC",
        "TSX": "^GSPTSE", "Crude": "CL=F", "Gold": "GC=F", "USDX": "DX-Y.NYB"
    }
    try:
        data = yf.download(list(indices.values()), period="2d", interval="1d", auto_adjust=True)
        close_data = data['Close'] if isinstance(data.columns, pd.MultiIndex) else data
        results = {}
        for name, sym in indices.items():
            if sym in close_data.columns:
                series = close_data[sym].dropna()
                if len(series) >= 2:
                    curr, prev = series.iloc[-1], series.iloc[-2]
                    diff = curr - prev
                    pct = (diff / prev) * 100
                    results[name] = {"val": curr, "pct": pct}
        return results
    except:
        return {}

# --- 3. Main Data Fetching ---
@st.cache_data(ttl=3600)
def fetch_financial_data(ticker, days):
    try:
        data = yf.download([ticker, "^VIX"], period=f"{days}d", interval="1d", auto_adjust=False)
        if isinstance(data.columns, pd.MultiIndex):
            prices = data['Adj Close']
        else:
            prices = data[['Adj Close']]
        return prices
    except:
        return pd.DataFrame()

def get_reddit_sentiment(ticker):
    return 0, 0

# --- 4. Sidebar Control ---
st.sidebar.title("Control Center")
ticker_input = st.sidebar.text_input("Ticker (e.g., AAPL 🇺🇸 | AC.TO 🇨🇦)", "AAPL").upper()

if ".TO" in ticker_input or ".V" in ticker_input:
    st.sidebar.success("Canadian Ticker Detected 🇨🇦")
    ticker = ticker_input
else:
    ticker = ticker_input
    if len(ticker_input) >= 2 and ticker_input.isalpha():
        st.sidebar.info("💡 Tip: For Canada, add .TO")

lookback = st.sidebar.slider("Lookback Period (Divergence)", 30, 250, 90)
st.sidebar.markdown("---")
# 这里的变量名 report_lang 必须对应你 352 行使用的名字
report_lang = st.sidebar.selectbox(" 🌐🇺🇸/🇨🇳", ["English", "中文"])


st.sidebar.markdown("---")
st.sidebar.caption("🚀  Designed by J")
st.sidebar.caption("🤖  Powered by Gemini AI")
st.sidebar.caption("📅  v3.0 | March 2026")


# --- 5. Market Index Bar Execution ---
index_data = fetch_market_indices()
st.title(f"📊 {ticker} Technical & Sentiment Dashboard")

if index_data:
    idx_cols = st.columns(len(index_data))
    for i, (name, d) in enumerate(index_data.items()):
        mode, arrow = "off", "•"
        if d['pct'] > 0: mode, arrow = "normal", "▲"
        elif d['pct'] < 0: mode, arrow = "inverse", "▼"
        idx_cols[i].metric(name, f"{d['val']:,.2f}", f"{arrow} {abs(d['pct']):.2f}%", delta_color=mode)
st.divider()

# --- 6. Main Indicators & Charts ---
prices = fetch_financial_data(ticker, lookback)

if not prices.empty and ticker in prices.columns:
    # Indicator Logic
    delta = prices[ticker].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rsi_series = 100 - (100 / (1 + (gain / loss)))
    
    current_vix = prices["^VIX"].iloc[-1] if "^VIX" in prices.columns else 0
    vix_sma = prices["^VIX"].rolling(20).mean().iloc[-1] if "^VIX" in prices.columns else 1

    # Metrics Section
    st.subheader(f"⚠️ {ticker} Real-time Sentiment Warning")
    mentions, wsb_score = get_reddit_sentiment(ticker)
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Price", f"${prices[ticker].iloc[-1]:.2f}")
    m2.metric("RSI", f"{rsi_series.iloc[-1]:.2f}", delta="OB" if rsi_series.iloc[-1] > 70 else "OS" if rsi_series.iloc[-1] < 30 else "Normal")
    m3.metric("VIX", f"{current_vix:.2f}", delta=f"{((current_vix/vix_sma)-1)*100:.1f}%", delta_color="inverse")
    m4.metric("WSB", f"{mentions}", delta="Sentiment Check")

    # Technical Chart
    st.subheader("📈 Technical Analysis (Bollinger + MACD)")
    daily = yf.download(ticker, period="1y", interval="1d")
    if isinstance(daily.columns, pd.MultiIndex): daily.columns = daily.columns.droplevel(1)
    
    ma20 = daily['Close'].rolling(20).mean()
    std20 = daily['Close'].rolling(20).std()
    up_bb, lo_bb = ma20 + (std20 * 2), ma20 - (std20 * 2)
    macd = daily['Close'].ewm(span=12).mean() - daily['Close'].ewm(span=26).mean()
    sig = macd.ewm(span=9).mean()
    hist = macd - sig

    apds = [
        mpf.make_addplot(up_bb, color='gray', alpha=0.2),
        mpf.make_addplot(lo_bb, color='gray', alpha=0.2),
        mpf.make_addplot(macd, panel=2, color='fuchsia', ylabel='MACD'),
        mpf.make_addplot(sig, panel=2, color='blue'),
        mpf.make_addplot(hist, panel=2, type='bar', color='gray', alpha=0.3)
    ]
    fig, axlist = mpf.plot(daily, type='candle', style='yahoo', volume=True, mav=(20, 50, 200), addplot=apds, panel_ratios=(6,2,2), returnfig=True, figsize=(12, 8))
    axlist[0].legend(['MA20', 'MA50', 'MA200', 'Upper BB', 'Lower BB'], loc='upper left', fontsize='x-small')
    st.pyplot(fig)

    # Divergence Chart
    st.divider()
    st.subheader("🔍 Price Momentum & Technical Divergence")
    fig_div, (ax_p, ax_r) = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    plt.subplots_adjust(hspace=0.1)
    ax_p.plot(prices.index, prices[ticker], color='#1f77b4', label="Price")
    ax_r.plot(rsi_series.index, rsi_series, color='#9467bd', label="RSI")
    ax_r.axhline(70, color='red', ls='--'); ax_r.axhline(30, color='green', ls='--')
    ax_p.legend(); ax_r.legend()
    st.pyplot(fig_div)
    
    
# --- 8. Chart Legend Expander (Professional Analysis) ---
    if report_lang == "English":
        with st.expander("📖 Professional Analysis: RSI & Volume Divergence"):
            st.markdown(f"""
            ### 1. RSI Divergence: Momentum Exhaustion
            RSI measures the 'speed' and 'strength' of price movements.

            #### **A. Bearish Divergence —— Exit Signal**
            * **Phenomenon**: Price hits a **new high**, but RSI line is trending **downward** (lower peak).
            * **Meaning**: Upward momentum is fading despite rising prices. Like a car sprinting on an empty tank.
            * **Action**: Consider reducing positions or raising stop-loss levels.

            #### **B. Bullish Divergence —— Buy Signal**
            * **Phenomenon**: Price hits a **new low**, but RSI line is trending **upward** (higher trough).
            * **Meaning**: Selling pressure is exhausting. 
            * **Action**: System detects **{ticker}** may be in this zone; a rebound is often imminent.

            ---

            ### 2. Volume Divergence: Capital Support
            Volume is the "fuel" of a stock. **Rising price with rising volume** is the healthiest trend.

            #### **A. Low Volume Rally —— False Prosperity**
            * **Meaning**: Buying power is depleted; usually retail chasing while institutions exit. High risk of sharp reversal.

            #### **B. High Volume Crash —— Panic Selling**
            * **Meaning**: Massive panic selling. If at the end of a downtrend, it signals a "washout"; if at a peak, it's a disaster.

            #### **C. Low Volume Pullback —— Consolidation**
            * **Meaning**: Selling is not aggressive; usually healthy profit-taking or institutional "shaking the tree".
            """)
    else:
        with st.expander("📖 核心技术指标深度解读：RSI 与 量价背离"):
            st.markdown(f"""
            ### 1. RSI 背离：判断“动力”是否衰竭
            RSI 衡量的是价格上涨或下跌的“速度”和“力度”。

            #### **A. 看跌背离 (Bearish Divergence) —— 逃顶信号**
            * **现象**：股价创出**新高**，但 RSI 线却在走**下坡路**（高点比前一个高点低）。
            * **含义**：虽然价格在涨，但支撑上涨的动能正在减弱。
            * **操作**：建议减仓或调高止损位。

            #### **B. 看涨背离 (Bullish Divergence) —— 抄底信号**
            * **现象**：股价创出**新低**，但 RSI 线却在走**上坡路**（低点比前一个低点高）。
            * **含义**：下跌的杀伤力已经减弱，空头力量正在衰竭。
            * **操作**：系统检测到 **{ticker}** 可能正处于此类信号中。
            """)

    # --- 9. Technical Indicators: Overbought, Oversold & Overextended ---
    if report_lang == "English":
        with st.expander("💡 Pro Guide: Identifying Overbought vs. Oversold"):
            st.markdown(f"""
            ### 1. Overbought —— Warning of Pullback
            * **Technical**: **RSI > 70** or price touching the **Upper Bollinger Band**.
            * **Strategy**: Take profit or reduce exposure; avoid chasing highs.

            ### 2. Oversold —— Watching for Rebound
            * **Technical**: **RSI < 30** or price piercing the **Lower Bollinger Band**.
            * **Strategy**: Potential buying opportunity; look for volume confirmation.

            ### 3. Overextended (Deep Value) —— Finding the Limit
            * **Definition**: Deeper than oversold; price is significantly below the 200-day MA.
            * **Technical**: **RSI < 20** and extreme negative Bias.
            * **Strategy**: High risk-reward ratio for "revenge rebounds."

            ---

            ### ⚠️ Professional Tips
            1. **Trend Trap**: In strong trends, RSI can stay overbought/oversold for a long time. 
            2. **Double Confirmation**: The signal is strongest when RSI crosses back inside the 30/70 levels.
            3. **Context**: If **VIX** is rising while **{ticker}** is oversold, the rebound probability increases.
            """)
    else:
        with st.expander("💡 进阶指南：如何识别超买、超卖与超跌"):
            st.markdown(f"""
            ### 1. 超买 (Overbought) —— 警惕回调
            * **技术识别**：**RSI > 70** 或股价触碰**布林带上轨**。
            * **操作策略**：通常是减仓信号，不建议此时追涨。

            ### 2. 超卖 (Oversold) —— 关注反弹
            * **技术识别**：**RSI < 30** 或股价穿出**布林带下轨**。
            * **操作策略**：潜在买入机会，需配合成交量确认。

            ### 3. 超跌 (Overextended) —— 寻找极限
            * **核心区别**：比超卖更严重，股价远低于 MA200 均线。
            * **操作策略**：极易引发“报复性反弹”。

            ---

            ### ⚠️ 交易员笔记 (Professional Tips)
            1. **趋势陷阱**：超买不代表立刻跌，超卖不代表立刻涨。
            2. **双重确认**：最可靠信号是 RSI 回到正常区间内。
            3. **结合背景**：系统检测 **{ticker}** 指标时，请同步关注 VIX 指数。
            """)
            
    # VIX & Earnings
    st.divider()
    vix_col, earn_col = st.columns([2, 1])
    with vix_col:
        st.subheader("📉 VIX Volatility Trend")
        vix_df = yf.download("^VIX", period=f"{lookback}d")
        if isinstance(vix_df.columns, pd.MultiIndex): vix_df.columns = vix_df.columns.droplevel(1)
        fig_v, ax_v = plt.subplots(figsize=(8, 3))
        ax_v.plot(vix_df.index, vix_df['Close'], color='red')
        ax_v.axhline(20, color='orange', ls='--')
        ax_v.fill_between(vix_df.index, vix_df['Close'], 20, where=(vix_df['Close'] > 20), color='red', alpha=0.1)
        st.pyplot(fig_v)

    # --- Integrated Tested Earnings Module ---
    with earn_col:
        st.subheader("📅 Earnings Warning")
        ticker_obj = yf.Ticker(ticker)
        next_earn_date = None
        try:
            earnings = ticker_obj.get_earnings_dates(limit=1)
            if not earnings.empty: next_earn_date = earnings.index[0].date()
            if next_earn_date is None:
                cal = ticker_obj.calendar
                if isinstance(cal, dict): next_earn_date = cal.get('Earnings Date')[0].date()
                elif isinstance(cal, pd.DataFrame): next_earn_date = cal.iloc[0, 0].date()
        except: pass

        if next_earn_date:
            days_left = (next_earn_date - datetime.now().date()).days
            st.info(f"Next Earnings: **{next_earn_date}** (In **{days_left}** days)")
            if 0 <= days_left <= 7: st.error("⚠️ Earnings Week: High IV expected!")
        else:
            st.warning("⚠️ Could not fetch earnings date automatically.")
        
        
        

    # --- Integrated Tested News Module ---
    st.divider()
    st.subheader(f"📰 {ticker} English Market News")

    @st.cache_data(ttl=600)
    def fetch_2026_news(symbol):
        news_items = []
        try:
            raw_yf = yf.Ticker(symbol).news
            for item in raw_yf[:5]:
                title = item.get('title') or item.get('headline') or (item.get('content', {}).get('title')) or "News Update"
                link = item.get('link') or item.get('url') or "https://finance.yahoo.com"
                ts = item.get('providerPublishTime') or item.get('pubDate')
                p_time = datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M') if isinstance(ts, int) else "Recently"
                news_items.append({'title': title, 'link': link, 'source': item.get('publisher') or "Yahoo", 'time': p_time})
        except: pass

        try:
            safe_q = urllib.parse.quote(f"{symbol} stock")
            rss_url = f"https://google.com{safe_q}&hl=en-US&gl=US"
            feed = feedparser.parse(rss_url)
            for e in feed.entries[:5]:
                news_items.append({'title': e.title, 'link': e.link, 'source': getattr(e, 'source', {}).get('title', 'Google News'), 'time': e.published})
        except: pass
        return news_items

    final_news = fetch_2026_news(ticker)
    if final_news:
        for item in final_news:
            with st.container():
                st.markdown(f"**[{item['title']}]({item['link']})**")
                st.caption(f"{item['source']} | {item['time']}")
                st.write("---")
    else:
        st.error("❌ Failed to retrieve news. Run: `pip install -U yfinance`.")


    # --- 10. Gemini AI 深度决策系统 ---
    st.divider()

    # 动态设置标题和按钮文字
    if report_lang == "English":
        header_text = "🤖 Activate Deep AI Decision System"
        button_text = "🚀 Generate Realtime AI Report"
        spinner_text = "Gemini is analyzing market data..."
    else:
        header_text = "🤖 启用 AI 深度决策系统"
        button_text = "🚀 生成实时 AI 报告"
        spinner_text = "Gemini 正在联网分析中..."

    st.header(header_text)

    # ... (中间的 tech_data 和 news_titles 定义保持不变) ...

    if st.button(button_text):
        with st.spinner(spinner_text):
            # 调用函数并传入 report_lang
            report = run_gemini_pro_analysis(ticker, tech_data, news_titles, report_lang)
            st.markdown(report)
    current_rsi_val = rsi_series.iloc[-1] if not rsi_series.empty else "N/A"
    tech_data = {
        "price": f"${prices[ticker].iloc[-1]:.2f}",
        "rsi": f"{current_rsi_val:.2f}",
        "vix": f"{current_vix:.2f}",
        "lookback": f"{lookback} days"
    }

    # 提取新闻标题列表
    news_titles = [item['title'] for item in final_news] if final_news else "No recent news found."




else:
    st.error("❌ Data Fetch Failed. Check connection or Ticker.")





