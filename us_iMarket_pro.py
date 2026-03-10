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
def run_gemini_pro_analysis(ticker, tech_metrics, news_summary):

    # ✅ 修改这一行：从 Streamlit 的安全设置里读取，而不是写死在代码里
    if "GEMINI_API_KEY" in st.secrets:
        api_key_val = st.secrets["GEMINI_API_KEY"]
    else:
        # 这里的报错会提醒你在后台配置 Key
        return "❌ 错误：未在 Streamlit Cloud 后台配置 GEMINI_API_KEY"
    
    genai.configure(api_key=api_key_val)
    
    try:
        # 1. 自动寻找你账号下可用的生成模型 (核心修复)
        available_models = [m.name for m in genai.list_models() if 'generateContent' in m.supported_generation_methods]
        
        if not available_models:
            return "❌ 你的 API Key 没有任何可用模型，请检查 Google AI Studio 的项目状态。"
        
        # 优先寻找 flash，其次寻找 pro，都没有就用第一个可用的
        target_model = ""
        for m in available_models:
            if 'flash' in m.lower():
                target_model = m
                break
        if not target_model:
            target_model = available_models[0]

        # 2. 使用自动找到的 target_model
        model = genai.GenerativeModel(target_model)

        prompt = f"""
        作为顶级华尔街量化策略师，请针对股票 {ticker} 撰写深度研究报告。
        【当前技术指标】: {tech_metrics}
        【近期市场新闻概要】: {news_summary}
        请按 Markdown 框架生成报告: 包含核心结论、技术面扫描、风险剖析及投资建议。
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
st.sidebar.caption("🚀 **Designed by J**")
st.sidebar.caption("🤖 **Powered by Gemini AI**")
st.sidebar.caption("📅 *v2.5 | March 2026*")

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

    # Expander
    # --- 8. Chart Legend Expander (Professional Analysis) ---
    with st.expander("📖 核心技术指标深度解读：RSI 与 量价背离"):
        st.markdown(f"""
        ### 1. RSI 背离：判断“动力”是否衰竭
        RSI 衡量的是价格上涨或下跌的“速度”和“力度”。

        #### **A. 看跌背离 (Bearish Divergence) —— 逃顶信号**
        *   **现象**：股价创出**新高**，但 RSI 线却在走**下坡路**（高点比前一个高点低）。
        *   **含义**：虽然价格在涨，但支撑上涨的动能正在减弱。这就像一辆赛车冲刺时燃油即将耗尽，虽然还在往前冲，但随时会熄火。
        *   **操作**：建议减仓或调高止损位。

        #### **B. 看涨背离 (Bullish Divergence) —— 抄底信号**
        *   **现象**：股价创出**新低**，但 RSI 线却在走**上坡路**（低点比前一个低点高）。
        *   **含义**：虽然价格在跌，但下跌的杀伤力已经减弱，空头力量正在衰竭。
        *   **操作**：系统检测到 **{ticker}** 可能正处于此类信号中，通常预示反弹即将来临。

        ---

        ### 2. 量价背离：判断“资金”是否支持
        成交量 (Volume) 是股票的“汽油”。**量价齐升**才是最健康的上涨。

        #### **A. 缩量上涨 (量价背离) —— 虚假繁荣**
        *   **现象**：价格持续上涨，但成交量却一天比一天小。
        *   **含义**：买入力量枯竭，通常是散户追涨而机构在悄悄出货。一旦买盘断档，股价极易剧烈回调。

        #### **B. 放量下跌 (量价同步) —— 恐慌杀跌**
        *   **现象**：价格大幅下跌，同时成交量剧增。
        *   **含义**：大量恐慌盘不计成本割肉。若出现在下跌末端，往往意味着“洗盘”结束；若出现在高位，则是趋势反转的灾难。

        #### **C. 缩量下跌 —— 阴跌/洗盘**
        *   **现象**：价格小幅下滑，但成交量很低。
        *   **含义**：卖盘并不积极，通常是正常的获利盘回吐或主力在震仓洗人。
        """)
    # --- 9. 技术指标深度说明：超买、超卖与超跌 ---
    with st.expander("💡 进阶指南：如何识别超买、超卖与超跌"):
        st.markdown(f"""
        ### 1. 超买 (Overbought) —— 警惕回调
        *   **定义**：指资产价格上涨过快、过高，超出了其内在价值或近期平均水平。
        *   **技术识别**：
            *   **RSI > 70**：反映上涨动能进入极端区域，买方力量可能接近衰竭。
            *   **布林带 (Bollinger Bands)**：股价触碰或穿出**上轨**，暗示价格偏离均值过远。
        *   **操作策略**：通常是**减仓或止盈**的信号，不建议此时追涨。

        ### 2. 超卖 (Oversold) —— 关注反弹
        *   **定义**：指资产价格下跌过快、过低，通常是由于市场情绪过度恐慌导致的非理性抛售。
        *   **技术识别**：
            *   **RSI < 30**：反映下跌动能进入极端区域，卖方力量可能已经透支。
            *   **布林带**：股价触碰或穿出**下轨**，暗示价格存在回归均值的需求。
        *   **操作策略**：通常是**潜在的买入机会**，但需配合成交量缩小或底背离来确认。

        ### 3. 超跌 (Overextended Downward) —— 寻找极限
        *   **定义**：超跌是比“超卖”更深层的状态，通常指股价经历了长时间或极大幅度的连续下跌。
        *   **核心区别**：超卖可能是短期的，而超跌往往意味着股价已经跌破了关键的长期支撑位（如 MA200 均线）。
        *   **技术识别**：
            *   **RSI 极低 (< 20)**：进入深度超卖区域。
            *   **乖离率 (Bias)**：股价大幅低于其长期移动平均线。
        *   **操作策略**：超跌股容易引发“报复性反弹”，是短线博弈高盈亏比机会的重点区域。

        ---

        ### ⚠️ 交易员笔记 (Professional Tips)
        1.  **趋势陷阱**：在强劲的单边趋势中，RSI 可以长时间停留在超买（或超卖）区。**“超买不代表立刻跌，超卖不代表立刻涨”**。
        2.  **双重确认**：最可靠的信号是当 RSI 从超买区**回落至 70 以下**，或从超卖区**回升至 30 以上**时。
        3.  **结合背景**：系统检测到 **{ticker}** 的指标时，务必结合大盘 VIX 恐慌指数。若 VIX 同步走高，超卖反弹的可靠性更强。
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
    st.header("🤖 Gemini AI 深度决策系统")
    
    current_rsi_val = rsi_series.iloc[-1] if not rsi_series.empty else "N/A"
    tech_data = {
        "price": f"${prices[ticker].iloc[-1]:.2f}",
        "rsi": f"{current_rsi_val:.2f}",
        "vix": f"{current_vix:.2f}",
        "lookback": f"{lookback} days"
    }
    
    # 提取新闻标题列表
    news_titles = [item['title'] for item in final_news] if final_news else "No recent news found."

    if st.button("🚀 生成实时综部分析报告"):
        with st.spinner("Gemini 正在联网分析中..."):
            # 直接调用上面定义的函数
            report = run_gemini_pro_analysis(ticker, tech_data, news_titles)
            st.markdown(report)

else:
    st.error("❌ Data Fetch Failed. Check connection or Ticker.")





