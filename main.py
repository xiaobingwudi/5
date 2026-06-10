"""
Al Brooks 日内机会寻找训练器 V5.0（完整版）
核心设计：一次性分析80根K线，输出完整故事
每根K线都有编号（从K1开始），用户可以精确引用
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import akshare as ak
from openai import OpenAI
import random
import time

# ==================== 配置 ====================
SYMBOL_NAMES = {
    "IF": "沪深300股指", "IH": "上证50股指", "IC": "中证500股指", "IM": "中证1000股指",
    "CU": "沪铜", "AL": "沪铝", "ZN": "沪锌", "PB": "沪铅", "NI": "沪镍", "SN": "沪锡",
    "AU": "黄金", "AG": "白银", "RB": "螺纹钢", "HC": "热轧卷板", "SS": "不锈钢",
    "I": "铁矿石", "J": "焦炭", "JM": "焦煤",
    "MA": "甲醇", "TA": "PTA", "SA": "纯碱", "FG": "玻璃",
    "A": "豆一", "M": "豆粕", "Y": "豆油", "P": "棕榈油", "C": "玉米",
    "CF": "棉花", "SR": "白糖", "AP": "苹果",
    "SC": "原油", "FU": "燃料油",
}

EXCHANGES = {
    "股指": ["IF", "IH", "IC", "IM"],
    "黑色": ["RB", "I", "J", "JM"],
    "化工": ["MA", "TA", "SA", "FG"],
    "农产品": ["A", "M", "Y", "P", "C", "CF", "SR"],
    "有色": ["CU", "AL", "ZN", "AU", "AG"],
    "能源": ["SC", "FU"],
}

# 五步框架
FIVE_STEPS = [
    {"name": "第1步：画线", "question": "画出通道和楔形。请引用具体K线编号说明你的画法和理由。"},
    {"name": "第2步：形态", "question": "识别双顶/双底、三角形等形态。请引用具体K线编号说明你看到了什么。"},
    {"name": "第3步：特殊K线", "question": "有哪些特殊K线（大K线、内包、外包、IOI等）？请引用具体K线编号。"},
    {"name": "第4步：值得做吗", "question": "这笔交易值得做吗？概率、空间、风险如何？请引用具体K线位置。"},
    {"name": "第5步：管理仓位", "question": "如果入场，如何管理？止损、目标、退出条件？请引用具体价格和K线。"},
]

# ==================== AI 提示词 ====================
STORY_ANALYSIS_SYSTEM = """你是 Al Brooks 价格行为教练。

【重要】用户看到的图表上，每根K线都有编号（K1, K2, K3...K80）。
用户必须引用具体的K线编号来说明他的观察。

【你的任务】
用户已经看到了完整的80根K线图表。
他正在按照五步框架，一次性分析整张图。
你的任务是：
1. 验证用户的故事是否完整
2. 检查用户是否引用了具体的K线编号
3. 追问遗漏的关键证据
4. 追问反证："什么情况会证明你的判断是错的？"

【输出格式】
- 先说肯定的话（"好的，我理解了"）
- 如果用户没有引用K线编号，请提醒他引用
- 然后问2-3个引导性问题
- 控制在150字以内

【图表信息】
K线范围: K1 ~ K{max_bar}（共{count}根K线）
整体方向: 从K1的{first_close:.0f}到K{max_bar}的{last_close:.0f}（{direction}）
整体波幅: {lowest:.0f} ~ {highest:.0f}（{range_pct:.1f}%）
"""


def call_story_analyzer(df, max_bar, user_story, conversation_history):
    """调用故事分析器"""
    api_key = st.secrets.get("OPENAI_API_KEY", "")
    base_url = st.secrets.get("OPENAI_BASE_URL", "https://api.deepseek.com")
    model = st.secrets.get("OPENAI_MODEL", "deepseek-chat")
    
    if not api_key:
        return "【提示】请配置API密钥\n\n请在 Streamlit Cloud 后台设置 OPENAI_API_KEY"
    
    first_close = df.iloc[0]["close"]
    last_close = df.iloc[max_bar - 1]["close"] if max_bar > 0 else df.iloc[0]["close"]
    highest = df.iloc[:max_bar]["high"].max()
    lowest = df.iloc[:max_bar]["low"].min()
    range_pct = (highest - lowest) / lowest * 100
    direction = "上涨" if last_close > first_close else "下跌" if last_close < first_close else "震荡"
    
    system = STORY_ANALYSIS_SYSTEM.format(
        max_bar=max_bar,
        count=max_bar,
        first_close=first_close,
        last_close=last_close,
        direction=direction,
        highest=highest,
        lowest=lowest,
        range_pct=range_pct
    )
    
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": f"用户的五步分析：\n{user_story}"}
    ]
    
    for msg in conversation_history[-6:]:
        if isinstance(msg, dict) and "role" in msg and "content" in msg:
            messages.append(msg)
    
    try:
        client = OpenAI(base_url=base_url, api_key=api_key)
        resp = client.chat.completions.create(
            model=model, messages=messages, temperature=0.3, max_tokens=400
        )
        return resp.choices[0].message.content
    except Exception as e:
        return f"AI错误: {str(e)[:100]}\n\n请检查API配置。"


# ==================== 数据加载 ====================
@st.cache_data(ttl=3600, show_spinner=False)
def load_data(symbol, period="30"):
    try:
        df = ak.futures_zh_minute_sina(symbol=symbol, period=period)
        if df is None or len(df) == 0:
            return None
        df = df.rename(columns={
            "date": "time", "open": "open", "high": "high",
            "low": "low", "close": "close", "volume": "volume"
        })
        return df.reset_index(drop=True)
    except Exception as e:
        st.error(f"数据加载失败: {e}")
        return None


# ==================== 图表绘制 ====================
def build_chart(df, max_bar):
    """
    绘制完整K线图，每根K线都有编号（从K1开始）
    max_bar: 显示的K线数量（例如80表示显示K1-K80）
    """
    # 取最近 max_bar 根K线
    start = max(0, len(df) - max_bar)
    plot_df = df.iloc[start:].copy().reset_index(drop=True)
    # 实际显示的K线数量
    n_bars = len(plot_df)
    # 编号从1开始：K1, K2, K3...
    bar_numbers = list(range(1, n_bars + 1))

    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        vertical_spacing=0.02, row_heights=[0.8, 0.2]
    )

    fig.add_trace(go.Candlestick(
        x=plot_df.index,
        open=plot_df["open"], high=plot_df["high"],
        low=plot_df["low"], close=plot_df["close"],
        showlegend=False,
        increasing_line_color="#ef5350",
        decreasing_line_color="#26a69a",
    ), row=1, col=1)

    vol_colors = ["#ef5350" if c >= o else "#26a69a"
                  for o, c in zip(plot_df["open"], plot_df["close"])]
    fig.add_trace(go.Bar(
        x=plot_df.index, y=plot_df["volume"],
        marker_color=vol_colors, showlegend=False, opacity=0.5
    ), row=2, col=1)

    # ========== 每根K线都标注编号（从K1开始）==========
    for idx, bar_num in enumerate(bar_numbers):
        row_data = plot_df.iloc[idx]
        # 根据K线方向决定标注位置
        if row_data["close"] >= row_data["open"]:
            # 阳线：编号标在K线下方
            y_pos = row_data["low"]
            y_shift = -12
        else:
            # 阴线：编号标在K线上方
            y_pos = row_data["high"]
            y_shift = 12
        
        fig.add_annotation(
            x=idx, y=y_pos,
            text=f"K{bar_num}",
            showarrow=False,
            font=dict(size=7, color="#888888"),
            yshift=y_shift,
            row=1, col=1
        )
    
    # 每隔10根K线用更大的字体突出显示
    for idx, bar_num in enumerate(bar_numbers):
        if bar_num % 10 == 0:
            row_data = plot_df.iloc[idx]
            if row_data["close"] >= row_data["open"]:
                y_pos = row_data["low"]
                y_shift = -22
            else:
                y_pos = row_data["high"]
                y_shift = 22
            fig.add_annotation(
                x=idx, y=y_pos,
                text=f"★ K{bar_num}",
                showarrow=False,
                font=dict(size=10, color="#333333", weight="bold"),
                yshift=y_shift,
                row=1, col=1
            )

    fig.update_layout(
        xaxis_rangeslider_visible=False,
        height=550,
        margin=dict(l=10, r=10, t=10, b=10),
        paper_bgcolor="#ffffff",
        plot_bgcolor="#f8f9fa",
        font=dict(color="#212529"),
    )
    fig.update_xaxes(showgrid=True, gridcolor="#e9ecef", gridwidth=0.5)
    fig.update_yaxes(showgrid=True, gridcolor="#e9ecef", gridwidth=0.5)
    
    # 添加说明文字
    fig.add_annotation(
        x=0.01, y=0.99, xref="paper", yref="paper",
        text="📌 每根K线都有编号（K1, K2, K3...）| ★标注每10根",
        showarrow=False,
        font=dict(size=10, color="#888888"),
        bgcolor="rgba(255,255,255,0.8)",
        borderpad=2
    )
    
    return fig


# ==================== Session State ====================
def init_state():
    defaults = {
        "df": None,
        "symbol": None,
        "max_bar": 80,  # 显示的K线数量
        "conversations": [],
        "analysis_complete": False,
        "practice_count": 0,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def reset_analysis():
    st.session_state.conversations = []
    st.session_state.analysis_complete = False


def load_new_symbol(code, period_value):
    with st.spinner(f"加载 {code} {period_value}分钟..."):
        df = load_data(f"{code}0", period=period_value)
        if df is None or len(df) < 100:
            st.error(f"{code} 数据加载失败")
            return False
        
        # 显示最近80根K线
        st.session_state.df = df
        st.session_state.symbol = code
        st.session_state.max_bar = 80
        reset_analysis()
        return True


# ==================== 主界面 ====================
def main():
    st.set_page_config(page_title="Al Brooks 五步训练器 V5.0", layout="wide")
    
    st.markdown("""
    <style>
    .stApp { background-color: #ffffff; }
    [data-testid="stSidebar"] {
        background-color: #f8f9fa !important;
        border-right: 1px solid #e9ecef;
    }
    .stExpander {
        background-color: #f8f9fa;
        border-radius: 8px;
        border: 1px solid #e9ecef;
    }
    .stChatMessage {
        background-color: #f8f9fa;
        border-radius: 12px;
        padding: 8px 12px;
    }
    </style>
    """, unsafe_allow_html=True)

    init_state()

    # 侧边栏
    with st.sidebar:
        st.markdown("### 📊 五步训练器 V5.0")
        st.markdown("---")
        st.markdown("**训练理念**")
        st.caption("一次性加载80根K线，一次性完成五步分析")
        st.caption("每根K线都有编号（K1-K80），分析时必须引用具体K线")
        st.caption("AI追问证据和反证，训练完整的故事阅读能力")
        
        st.markdown("---")
        st.metric("完成复盘次数", st.session_state.practice_count)
        
        st.markdown("---")
        period_map = {"15分钟": "15", "30分钟": "30", "60分钟": "60"}
        period = st.selectbox("周期", list(period_map.keys()), index=1)
        period_value = period_map[period]

        st.markdown("**选择品种**")
        for cat, codes in EXCHANGES.items():
            with st.expander(cat):
                cols = st.columns(2)
                for idx, code in enumerate(codes):
                    name = SYMBOL_NAMES.get(code, code)
                    if cols[idx % 2].button(f"{code}\n{name}", key=f"btn_{code}", use_container_width=True):
                        if load_new_symbol(code, period_value):
                            st.rerun()
        
        st.markdown("---")
        if st.session_state.df is not None:
            if st.button("🔄 重置当前分析", use_container_width=True):
                reset_analysis()
                st.rerun()
            
            total_bars = len(st.session_state.df)
            st.caption(f"数据总K线: {total_bars}根 | 显示最新80根")

    # 主界面
    if st.session_state.df is None:
        st.markdown("## 👈 请从左侧选择品种开始训练")
        st.markdown("""
        <div style="background:#f8f9fa;padding:20px;border-radius:12px;border:1px solid #e9ecef;">
        <h3>Al Brooks 五步训练器 V5.0</h3>
        <p><strong>一次性分析80根K线，输出完整故事。</strong></p>
        <p>训练流程：</p>
        <ol>
            <li>加载80根K线图表（每根K线都有编号<strong>K1, K2, K3...K80</strong>）</li>
            <li>按照五步框架，一次性分析整张图，<strong>必须引用具体K线编号</strong></li>
            <li>AI会检查你是否引用了K线编号，并追问证据和反证</li>
            <li>完成分析后，系统记录一次完整复盘</li>
        </ol>
        <p style="color:#6c757d;">💡 五步框架：画线 → 形态 → 特殊K线 → 值得做吗 → 管理仓位</p>
        </div>
        """, unsafe_allow_html=True)
        return

    df = st.session_state.df
    max_bar = st.session_state.max_bar

    # 显示图表
    st.plotly_chart(build_chart(df, max_bar), use_container_width=True)

    # 显示五步参考
    with st.expander("📖 五步框架参考", expanded=False):
        for step in FIVE_STEPS:
            st.markdown(f"**{step['name']}**")
            st.caption(step['question'])
            st.markdown("---")

    # 对话区域
    conv = st.session_state.conversations
    
    for msg in conv:
        if isinstance(msg, dict) and "role" in msg and "content" in msg:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

    # 分析完成
    if st.session_state.analysis_complete:
        st.success("🎉 完成一次完整复盘！")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🔄 开始下一次复盘", type="primary"):
                st.session_state.practice_count += 1
                reset_analysis()
                st.rerun()
        with col2:
            if st.button("🎲 换一个品种"):
                all_codes = []
                for codes in EXCHANGES.values():
                    all_codes.extend(codes)
                new_code = random.choice(all_codes)
                load_new_symbol(new_code, "30")
                st.rerun()
        return

    # 等待用户输入
    st.markdown("### ✏️ 你的分析")
    st.caption("请按照五步框架，引用具体K线编号（如K15、K32）描述你的观察")
    
    user_input = st.chat_input("例如：K5-K15形成下降通道，K28是双底右底...")

    if user_input:
        conv.append({"role": "user", "content": user_input})
        
        with st.spinner("AI分析中..."):
            ai_response = call_story_analyzer(df, max_bar, user_input, conv)
        
        conv.append({"role": "assistant", "content": ai_response})
        
        # 判断是否完成：输入超过150字且引用了K线编号
        import re
        has_k_line = bool(re.search(r'K\d+', user_input))
        if len(user_input) > 150 and has_k_line:
            st.session_state.analysis_complete = True
        
        st.rerun()


if __name__ == "__main__":
    main()
