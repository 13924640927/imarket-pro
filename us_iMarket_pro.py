import streamlit as st
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import mplfinance as mpf
import feedparser
import urllib.parse
from datetime import datetime
import google.generativeai as genai # 添加这一行
import numpy as np
# --- 1. Basic Configuration ---
st.set_page_config(
    page_title="iMarket Professional | AI-Quant Terminal", 
    page_icon="⚖️", # 或者 "📈"
    layout="wide"
    )

# --- 新增：深度估值计算函数 ---
def get_advanced_valuation(ticker, discount_rate=0.15):
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        fcf = info.get('freeCashflow', 0)
        shares = info.get('sharesOutstanding', 1)
        curr_price = info.get('currentPrice', 1)
        
        # 保守型 DCF 计算 (5年期)
        growth_rate = 0.05 
        pv_fcf = sum([fcf * (1 + growth_rate)**i / (1 + discount_rate)**i for i in range(1, 6)])
        terminal_v = (fcf * (1 + growth_rate)**5 * 1.02) / (discount_rate - 0.02)
        pv_tv = terminal_v / (1 + discount_rate)**5
        dcf_fair_value = (pv_fcf + pv_tv) / shares if shares > 0 else 0
        upside = (dcf_fair_value / curr_price - 1) * 100

        # 相对估值
        ev_sales = info.get('enterpriseToRevenue', 0)
        gp = info.get('grossProfits', 1)
        ev_gp = info.get('enterpriseValue', 0) / gp if gp != 0 else 0
        
        return {
            "dcf_price": dcf_fair_value,
            "upside_pct": upside,
            "ev_sales": ev_sales,
            "ev_gp": ev_gp,
            "sector": info.get('sector', 'N/A'),
            "industry_avg_s": 4.5 # 示例行业基准
        }
    except:
        return None

# --- 新增：模型专用 AI 分析函数 ---
def run_valuation_model_analysis(ticker, val_data, lang):
    # 1. 密钥检查与初始化 (代码保持不变)
    if "GEMINI_API_KEY" not in st.secrets: return "❌ API Key Missing"
    genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
    model = genai.GenerativeModel('models/gemini-3-flash-preview')

    # 2. 核心修复：根据语言完全隔离指令，不混用中英文
    if lang == "English":
        format_cmd = """
        IMPORTANT FORMATTING RULES:
        1. Add a space BETWEEN numbers and text.
        2. Use TWO newlines between bullet points to prevent text clumping.
        3. Bold all key metrics (e.g., **63.84**).
        """
        prompt = f"""
        Role: Senior Equity Analyst.
        Data for {ticker}:
        - DCF: ${val_data['dcf_price']:.2f} (Upside: {val_data['upside_pct']:.1f}%)
        - EV/Sales: {val_data['ev_sales']:.2f}x
        - EV/GP: {val_data['ev_gp']:.2f}x
        
        Task: Analyze if this is a 'Golden Pit' or 'Value Trap'. 
        {format_cmd}
        Output Language: Strictly English.
        """
    else:
        # 中文版：确保指令全是中文，防止 AI 跑偏
        format_cmd = """
        排版要求：
        1. 数字与汉字之间必须保留一个空格。
        2. 每个要点（Bullet Point）之间必须使用两个换行符，确保视觉上完全分开。
        3. 加粗关键指标（例如：**63.84**）。
        """
        prompt = f"""
        角色：资深量化分析师。
        {ticker} 建模数据：
        - DCF 内在价值: ${val_data['dcf_price']:.2f} (空间: {val_data['upside_pct']:.1f}%)
        - EV/Sales: {val_data['ev_sales']:.2f}x
        - EV/GP: {val_data['ev_gp']:.2f}x
        
        任务：判断该公司是“黄金坑”还是“估值陷阱”。
        {format_cmd}
        输出语言：必须使用中文。
        """

    try:
        response = model.generate_content(prompt)
        # 3. 终极后处理：代码层强制替换，确保 Markdown 换行生效
        cleaned_text = response.text.replace("\n*", "\n\n*").replace("\n-", "\n\n-")
        return cleaned_text
    except Exception as e:
        return f"AI Error: {str(e)}"


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
st.sidebar.image("iMarket Pro.png", use_container_width=True)
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


# --- 品牌标题：位置上移并微调黑字副标题 ---
st.markdown(
    """
    <div style="text-align: center; margin-top: -40px; margin-bottom: 5px; padding-top: 0px;">
        <h1 style="
            font-family: 'Segoe UI', Roboto, Helvetica, Arial, sans-serif; 
            font-weight: 800; 
            background: linear-gradient(135deg, #d4af37 25%, #f7e7ce 50%, #d4af37 75%); 
            -webkit-background-clip: text; 
            -webkit-text-fill-color: transparent; 
            text-shadow: 1px 1px 8px rgba(212, 175, 55, 0.2);
            letter-spacing: -1px; 
            margin-bottom: 0px;
            font-size: 3.2rem;
            line-height: 1.1;
        ">
            iMarket Pro
        </h1>
        <p style="
            color: #000000; 
            font-size: 1.1rem; 
            font-weight: 600; 
            letter-spacing: 2px; 
            text-transform: uppercase;
            margin-top: -8px;
            margin-bottom: 10px;
        ">
            AI-Powered Market Research Engine
        </p>
    </div>
    """, 
    unsafe_allow_html=True
)

index_data = fetch_market_indices()
# st.title(f"📊 {ticker} Technical & Sentiment Dashboard")






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
    # --- 指标逻辑计算 ---
    delta_series = prices[ticker].diff()
    gain = (delta_series.where(delta_series > 0, 0)).rolling(14).mean()
    loss = (-delta_series.where(delta_series < 0, 0)).rolling(14).mean()
    rsi_series = 100 - (100 / (1 + (gain / loss)))
    
    current_vix = prices["^VIX"].iloc[-1] if "^VIX" in prices.columns else 0
    vix_sma = prices["^VIX"].rolling(20).mean().iloc[-1] if "^VIX" in prices.columns else 1

    # --- 新增：Price 涨跌核心逻辑 ---
    curr_price = prices[ticker].iloc[-1]
    prev_close = prices[ticker].iloc[-2]
    price_change = curr_price - prev_close
    price_change_pct = (price_change / prev_close) * 100

    # --- Metrics Section (仪表盘指标卡) ---
    st.subheader(f"⚠️ {ticker} Real-time Sentiment Warning")
    mentions, wsb_score = get_reddit_sentiment(ticker)
    
    m1, m2, m3, m4 = st.columns(4)
    
    # 替换后的 Price：现在具备涨跌色块和标识
    m1.metric(
        label="Price", 
        value=f"${curr_price:.2f}", 
        delta=f"{price_change:+.2f} ({price_change_pct:+.2f}%)"
    )
    
    # RSI 保持原样
    m2.metric(
        label="RSI", 
        value=f"{rsi_series.iloc[-1]:.2f}", 
        delta="OB" if rsi_series.iloc[-1] > 70 else "OS" if rsi_series.iloc[-1] < 30 else "Normal"
    )
    
    # VIX 保持原样
    m3.metric(
        label="VIX", 
        value=f"{current_vix:.2f}", 
        delta=f"{((current_vix/vix_sma)-1)*100:.1f}%", 
        delta_color="inverse"
    )
    
    # WSB 保持原样
    m4.metric(
        label="WSB", 
        value=f"{mentions}", 
        delta="Sentiment Check"
    )
    
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

# --- Integrated Tested Earnings Module (V3 Refined) ---
    with earn_col:
        ticker_obj = yf.Ticker(ticker)
        next_earn_date = None
        
        try:
            # 1. 优先使用新版接口
            earnings = ticker_obj.get_earnings_dates(limit=1)
            if earnings is not None and not earnings.empty:
                next_earn_date = earnings.index[0].date()
            
            # 2. 如果失败，尝试备选 calendar 接口
            if next_earn_date is None:
                cal = ticker_obj.calendar
                if isinstance(cal, dict) and 'Earnings Date' in cal:
                    next_earn_date = cal.get('Earnings Date')[0].date()
                elif isinstance(cal, pd.DataFrame) and not cal.empty:
                    next_earn_date = cal.iloc[0, 0].date()
        except:
            pass # 失败时不报错，静默处理

        # --- 核心改进：只有拿到日期才显示 UI ---
        if next_earn_date:
            st.subheader("📅 Earnings Schedule") # 只有成功了才显示标题
            days_left = (next_earn_date - datetime.now().date()).days
            
            if days_left >= 0:
                st.info(f"Next Earnings: **{next_earn_date}** (In **{days_left}** days)")
                if 0 <= days_left <= 7:
                    st.error("⚠️ Earnings Week: High IV expected!")
            else:
                # 日期已过（比如是昨天），显示为“待定”
                st.caption("📅 Next Earnings: TBA (Post-Earnings Period)")
        else:
            # 自动隐藏警告：如果获取不到，直接不占地方，或者只留一个极小的灰色提示
            # 删掉原本的 st.warning，改为下面这种不显眼的提示
            st.caption("📅 Earnings info currently unavailable from Yahoo Finance")
            
        
        
        

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


    # --- 10. Gemini AI 深度决策系统 (双引擎版) ---
    st.divider()

    # 【1. 数据准备区】：确保所有按钮都能拿到最新的数据
    current_rsi_val = rsi_series.iloc[-1] if not rsi_series.empty else "N/A"
    tech_data = {
        "price": f"${prices[ticker].iloc[-1]:.2f}",
        "rsi": f"{current_rsi_val:.2f}",
        "vix": f"{current_vix:.2f}",
        "lookback": f"{lookback} days"
    }
    news_titles = [item['title'] for item in final_news] if final_news else "No recent news found."

    # 【2. 动态 UI 文字配置】
    if report_lang == "English":
        h_text = "🤖 AI Decision Dual Engines"
        b1_text = "🚀 Real-time AI Report"
        b2_text = "💎 Deep Valuation Model"
        s1_text = "Analyzing Technicals & News..."
        s2_text = "Running DCF & Valuation Models..."
    else:
        h_text = "🤖 AI 决策双引擎系统"
        b1_text = "🚀 生成实时 AI 报告"
        b2_text = "💎 运行深度估值模型"
        s1_text = "正在联网分析技术面与新闻..."
        s2_text = "正在运行 DCF 与行业对比模型..."

    st.header(h_text)

    # 【3. 按钮布局区】：并列展示
    col_btn1, col_btn2 = st.columns(2)

    with col_btn1:
        # 按钮 1：你原来的老功能，完全不动核心逻辑
        if st.button(b1_text, use_container_width=True):
            with st.spinner(s1_text):
                report = run_gemini_pro_analysis(ticker, tech_data, news_titles, report_lang)
                st.markdown(report)

    with col_btn2:
        # 按钮 2：新增的模型功能
        if st.button(b2_text, use_container_width=True):
            with st.spinner(s2_text):
                # 获取财务模型数据
                v_data = get_advanced_valuation(ticker, 0.15) # 0.15 是高折现率
                if v_data:
                    # 展现核心数据卡片
                    m1, m2, m3 = st.columns(3)
                    m1.metric("DCF Value", f"${v_data['dcf_price']:.2f}", f"{v_data['upside_pct']:.1f}%")
                    m2.metric("EV/Sales", f"{v_data['ev_sales']:.2f}x")
                    m3.metric("EV/GP", f"{v_data['ev_gp']:.2f}x")
                    
                    # 调用专门的模型分析函数（这个函数需要你在代码前面定义好）
                    model_report = run_valuation_model_analysis(ticker, v_data, report_lang)
                    st.info(model_report)
                else:
                    st.error("Financial data unavailable for this ticker.")

    with st.expander("📖 核心估值模型深度解读：DCF 与 企业价值倍数" if report_lang=="中文" else "📖 Deep Dive: DCF & Valuation Multiples"):
        if report_lang == "中文":
            st.markdown("""
            ### 1. DCF (贴现现金流) - 寻找内在价值
            * **原理**：DCF 认为公司现在的价值等于它未来能赚到的所有钱“折现”到今天的总和。
            * **高折现率策略**：本系统默认采用 **15% 折现率**。这是一个极度保守的“滤网”，只有当股价远低于这个标准时，才具有真正的**安全边际**。
            
            ### 2. EV/Sales (企业价值/销售额) - 规模与定价权
            * **逻辑**：相比 P/S，EV 考虑了公司的负债。
            * **判读**：如果该指标显著低于行业平均，可能存在**低估**；如果极高且缺乏增长支撑，则是**估值泡沫**。

            ### 3. EV/Gross Profit (企业价值/毛利) - 护城河指标
            * **核心**：这是衡量 AI 与软件公司最硬核的指标。它反映了公司每 1 元毛利在市场上被赋予的溢价。
            * **百分位意义**：查看当前倍数在过去 5 年的位置。处于 **20% 分位以下** 通常意味着处于“历史性底部”。
            """)
        else:
            st.markdown("""
            ### 1. DCF (Discounted Cash Flow) - The Intrinsic Value
            * **Principle**: DCF posits that a company is worth the sum of all its future cash flows, brought back to present value.
            * **High Discount Rate**: We use a **15% Discount Rate** by default. This acts as a conservative filter, ensuring a significant **Margin of Safety**.
            
            ### 2. EV/Sales - Scale & Pricing Power
            * **Logic**: Unlike P/S, EV (Enterprise Value) accounts for the company's debt and cash levels.
            * **Interpretation**: Significantly lower than industry average suggests **undervaluation**; excessively high suggests a **valuation bubble**.

            ### 3. EV/Gross Profit - The Moat Metric
            * **Core**: The ultimate metric for AI & SaaS firms. It shows the premium the market pays for every $1 of gross profit.
            * **Percentile**: Metrics below the **20th percentile** over 5 years often indicate a "Historical Floor."
            """)
    with st.expander("💡 进阶指南：如何区分“黄金坑”与“估值陷阱”" if report_lang=="中文" else "💡 Advanced Guide: Golden Pit vs. Value Trap"):
        if report_lang == "中文":
            st.info("""
            **🔍 识别黄金坑 (Golden Pit)**
            - **指标**：DCF 空间 > 20% 且 EV/GP 处于历史低位。
            - **信号**：AI 报告中提到“利空出尽”、“基本面改善”或“机构暗中吸筹”。
            
            **⚠️ 警惕估值陷阱 (Value Trap)**
            - **指标**：估值看起来极低，但 DCF 计算显示未来现金流正在萎缩。
            - **信号**：新闻中频繁出现“裁员”、“核心技术流失”或“法律诉讼”。
            """)
        else:
            st.info("""
            **🔍 Identifying a Golden Pit**
            - **Metrics**: DCF Upside > 20% and EV/GP at historical lows.
            - **Signals**: AI report mentions "Negative news priced in" or "Fundamental turnaround."
            
            **⚠️ Beware of Value Traps**
            - **Metrics**: Ratios look cheap, but DCF reveals shrinking future cash flows.
            - **Signals**: Frequent news regarding "Layoffs," "Loss of key talent," or "Litigation."
            """)
            
else:
    st.error("❌ Data Fetch Failed. Check connection or Ticker.")





