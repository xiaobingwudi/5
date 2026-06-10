"""
Al Brooks 日内机会寻找训练器 V4.1
视觉优化版

优化内容：
1. 侧边栏改为更舒适的配色（深蓝灰背景 + 高对比文字）
2. 整体界面更柔和，减少视觉疲劳
3. 调整字体颜色和对比度
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

MAX_ROUNDS_PER_STEP = 2

# ==================== 5步流程定义 ====================
STEPS = {
    1: {
        "name": "第1步：画线",
        "title": "寻找通道、楔形和超出(Overshoot)",
        "question": "你会如何画线？请说明你的选择和理由。",
        "core_elements": [
            "主趋势线（连接主要高点或低点）",
            "平行通道线（与主趋势线平行的支撑/阻力线）",
            "超出(Overshoot)：是否有K线突破了你画的线？",
            "备选画法：是否存在其他合理的画线方式？",
            "最终选择：你选择哪条线？为什么？"
        ],
        "example_good": "我用K5、K15、K25的高点画了一条下降趋势线。K8、K18的低点构成了平行支撑线，形成一个下降通道。K18的低点向下刺穿了支撑线形成Overshoot，但很快收回。我也考虑过连接K5和K25，但K15的超出更明显，所以我选择了更陡峭的那条。",
        "example_bad": "这是一个下降通道。",
    },
    2: {
        "name": "第2步：形态",
        "title": "识别双顶/双底、三角形等反转形态",
        "question": "图表上出现了哪些反转或延续形态？",
        "core_elements": [
            "双顶或双底（等高、更高/更低高点、更高/更低低点）",
            "微型双顶/双底（2-4根K线内形成）",
            "楔形（上升/下降，通常包含在通道内）",
            "三角形（收缩型/扩展型）",
            "主趋势反转(MTR)：突破重要趋势线",
            "失败突破（假突破）",
        ],
        "example_good": "K12和K28构成一个更高低点的双底，右底更强。K15-K20之间有一个收缩三角形，低点K17比K15高，高点K19比K16低。K22向上突破失败，形成假突破，立刻被空头打压。",
        "example_bad": "这里有双底，所以一定会上涨。这里有楔形，所以马上反转。这里突破了，所以必须追。",
    },
    3: {
        "name": "第3步：特殊K线",
        "title": "识别信号K线 (IB, OB, IOI, 大K线, 连续强势K线)",
        "question": "有哪些特殊的单根K线或K线组合？",
        "core_elements": [
            "大K线（实体占波幅>80%），以及它的收盘位置",
            "内包K线(IB)",
            "外包K线(OB)",
            "IOI模式（外包->内包->外包）",
            "连续内包线或连续外包线",
            "连续同向强势K线（3根以上，且重叠很少）",
            "高位收盘/低位收盘",
        ],
        "example_good": "K15是大阳线，实体92%，高位收盘，表明买方强劲。K16是内包线。K17是外包线，包含K16，形成IOI模式，预示着即将突破。K33-K35是三根连续的多头K线，几乎没有重叠，说明市场持续强劲买入。",
        "example_bad": "K15是大K线，很强势。",
    },
    4: {
        "name": "第4步：值得做吗",
        "title": "评估风险回报，决定是否入场",
        "question": "这笔交易值得做吗？",
        "core_elements": [
            "止损位置：具体价格（通常设在信号K线高/低点外侧）",
            "风险距离：从入场价到止损的距离",
            "第一目标位置（至少是风险的2倍以上）",
            "风险回报比(R:R)至少要达到2:1",
            "如果不值得，放弃；如果值得，怎么进场？"
        ],
        "example_good": "做多止损设在K28低点下方1 tick（3540），入场价约3560，风险20点。第一目标3600（2R），风险回报比2:1，值得做。采用突破单入场，在K28高点+1 tick。",
        "example_bad": "在K28做多，感觉要涨。",
    },
    5: {
        "name": "第5步：管理仓位",
        "title": "持仓期间的管理计划",
        "question": "入场后如何管理这笔交易？",
        "core_elements": [
            "仓位规模：根据风险决定（通常每笔风险1-2%）",
            "实现概率：这笔交易成功的概率有多高？(高/中/低)",
            "提前退出条件：什么情况下会提前离场？",
            "移动止损：什么情况下移动止损到保本？",
            "分批止盈计划"
        ],
        "example_good": "仓位规模为总资金的2%。成功概率高，因为它结合了形态、K线和通道的多重信号。若入场后价格在K28收盘价下方持续3根K线，则提前离场。涨到3600后移动止损到入场价保本。",
        "example_bad": "拿着等涨就行。",
    },
}

# ==================== 结构分析函数（后台使用，用户不可见）====================
def find_swing_points_in_range(df, start_idx, end_idx, pivot_window=2):
    sub = df.iloc[start_idx:end_idx + 1].copy().reset_index(drop=True)
    n = len(sub)
    w = pivot_window
    swing_highs = []
    swing_lows = []
    
    if n < w * 2 + 1:
        return swing_highs, swing_lows
    
    for i in range(w, n - w):
        h = sub.iloc[i]["high"]
        l = sub.iloc[i]["low"]
        orig_idx = start_idx + i
        
        left_higher = all(sub.iloc[i - j]["high"] < h for j in range(1, w + 1))
        right_higher = all(sub.iloc[i + j]["high"] < h for j in range(1, w + 1))
        if left_higher and right_higher:
            swing_highs.append({"idx": orig_idx, "price": h})
        
        left_lower = all(sub.iloc[i - j]["low"] > l for j in range(1, w + 1))
        right_lower = all(sub.iloc[i + j]["low"] > l for j in range(1, w + 1))
        if left_lower and right_lower:
            swing_lows.append({"idx": orig_idx, "price": l})
    
    return swing_highs, swing_lows


def detect_bar_patterns_in_range(df, start_idx, end_idx, lookback=50):
    actual_start = max(start_idx, end_idx - lookback)
    sub = df.iloc[actual_start:end_idx + 1].copy().reset_index(drop=True)
    n = len(sub)
    patterns = []

    for i in range(1, n):
        prev = sub.iloc[i - 1]
        curr = sub.iloc[i]
        orig_idx = actual_start + i
        ph, pl = prev["high"], prev["low"]
        ch, cl = curr["high"], curr["low"]
        co, cc = curr["open"], curr["close"]
        body = abs(cc - co)
        total = ch - cl
        body_ratio = body / total if total > 0 else 0

        if ch < ph and cl > pl:
            patterns.append(f"K{orig_idx}: 内包线")
        elif ch > ph and cl < pl:
            patterns.append(f"K{orig_idx}: 外包线")

        if body_ratio > 0.80:
            direction = "阳线" if cc > co else "阴线"
            patterns.append(f"K{orig_idx}: 大{direction}({body_ratio*100:.0f}%实体)")

    return patterns[-15:]


# ==================== AI 提示词 ====================
COACH_SYSTEM = """你是 Al Brooks 价格行为分析教练。

【你的角色】
- 你正在训练一个交易员养成"每天按固定流程复盘"的习惯。
- 你不是考官，你是陪练。
- 目标不是让用户"答对"，而是帮他"逐步形成自己的观察框架"。

【如何引导】
- 优先观察用户是否遗漏了当前步骤最核心的几个要素。
- 鼓励他引用具体K线编号、价格、形态。

【画线灵活性原则 - 极其重要】
- 双顶/双底很少完美，不要求精准。
- 通道画法可以有多种，关键是能否解释得通。
- **如果用户的画线方式与你参考的结构不同：只要逻辑成立，绝对不要纠正成后台画法。**
- 优先询问："为什么你更喜欢这种画法？" 而不是 "正确画法是什么？"

【禁止预测 - 极其重要】
- 发现形态 ≠ 预测结果。
- 如果用户把形态直接等同于结果（如"双底所以一定涨"），请提醒：
  "这个形态说明了什么？而不是一定会发生什么？"

【输出要求】
- 反馈请控制在120字以内。
- 每轮对话后，如果用户已经覆盖了核心要素，输出"[NEXT]"标记
- 如果还需要补充，输出"[CONTINUE]"标记

【当前训练步骤】
{step_info}

【用户历史薄弱点】
{profile_text}

【前几步用户发现】
{previous_findings}
"""


def build_step_info(step_num):
    step = STEPS[step_num]
    lines = [
        f"步骤：{step['name']} - {step['title']}",
        f"问题：{step['question']}",
        "在这个步骤中，可以观察：",
    ]
    for ce in step["core_elements"]:
        lines.append(f"  - {ce}")
    lines.append(f"好的示例：{step['example_good']}")
    lines.append(f"不好的示例：{step['example_bad']}")
    return "\n".join(lines)


def build_profile_text(reading_profile):
    if not reading_profile:
        return "暂无"
    items = []
    for step_key, weaknesses in reading_profile.items():
        for w, count in weaknesses.items():
            if count >= 2:
                items.append(f"第{step_key}步常漏：{w}")
    return "\n".join(items) if items else "无明显薄弱点"


def build_previous_findings(step_summaries):
    if not step_summaries:
        return "这是第1步，无前置发现。"
    return "\n".join(f"第{k}步发现：{v}" for k, v in step_summaries.items())


# ==================== AI 调用函数（适配您的配置）====================
def _gpt(messages):
    """调用 DeepSeek API（适配您的 Streamlit Cloud 配置）"""
    api_key = st.secrets.get("OPENAI_API_KEY", "")
    base_url = st.secrets.get("OPENAI_BASE_URL", "https://api.deepseek.com")
    model = st.secrets.get("OPENAI_MODEL", "deepseek-chat")
    if not api_key:
        st.error("API 密钥未配置！请在 Streamlit Cloud 后台设置 OPENAI_API_KEY")
        return "【配置提示】请先配置 DeepSeek API 密钥后再开始训练。"
    try:
        client = OpenAI(base_url=base_url, api_key=api_key)
        resp = client.chat.completions.create(
            model=model, messages=messages, temperature=0.2, max_tokens=700
        )
        return resp.choices[0].message.content
    except Exception as e:
        st.error(f"API 调用失败: {e}")
        return f"【API 错误】{e}"


def call_coach(step_num, conversation_history, structure_report, detected_elements, 
               step_summaries, reading_profile, round_num, max_rounds):
    """调用教练 API"""
    system_prompt = COACH_SYSTEM.format(
        structure_report=structure_report,
        detected_elements="、".join(detected_elements) if detected_elements else "无明显形态",
        step_info=build_step_info(step_num),
        profile_text=build_profile_guidance(reading_profile),
        previous_findings=build_previous_findings(step_summaries),
    )
    messages = [{"role": "system", "content": system_prompt}] + conversation_history
    
    response = _gpt(messages)
    
    # 确保返回格式包含标记
    if "[NEXT]" not in response and "[CONTINUE]" not in response:
        if round_num >= max_rounds:
            return response + "\n[NEXT]"
        return response + "\n[CONTINUE]"
    return response


def summarize_step(step_num, user_answers_text):
    """生成本步骤摘要"""
    system = "用30字以内总结用户在本步骤的关键发现，只输出发现内容，不要评价。"
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user_answers_text}
    ]
    return _gpt(messages)


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
    except Exception:
        return None


# ==================== 图表绘制 ====================
def build_chart(df, bar, step_name=""):
    end = bar + 1
    start = max(0, end - 80)
    plot_df = df.iloc[start:end].copy().reset_index(drop=True)
    original_indices = list(range(start, end))

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

    for idx, orig_idx in enumerate(original_indices):
        if orig_idx % 5 == 0:
            row_data = plot_df.iloc[idx]
            fig.add_annotation(
                x=idx, y=row_data["low"],
                text=str(orig_idx), showarrow=False,
                font=dict(size=7, color="#888888"),
                yshift=-14, row=1, col=1
            )

    bar_pos = original_indices.index(bar) if bar in original_indices else len(original_indices) - 1
    fig.add_vline(x=bar_pos, line_dash="dash", line_color="#ff9800", line_width=1.5, opacity=0.8)

    if step_name:
        fig.add_annotation(
            x=0.02, y=0.98, xref="paper", yref="paper",
            text=f"📌 {step_name}", showarrow=False,
            font=dict(size=12, color="#ffffff"),
            bgcolor="rgba(0,0,0,0.6)", borderpad=4
        )

    fig.update_layout(
        xaxis_rangeslider_visible=False,
        height=480,
        margin=dict(l=5, r=5, t=5, b=5),
        paper_bgcolor="#0e1117",
        plot_bgcolor="#1a1a2e",
        font=dict(color="#e0e0e0"),
    )
    fig.update_xaxes(showgrid=True, gridcolor="#2d3561", gridwidth=0.5)
    fig.update_yaxes(showgrid=True, gridcolor="#2d3561", gridwidth=0.5)
    return fig


# ==================== Session State ====================
def init_state():
    defaults = {
        "step": 1,
        "current_bar": 80,
        "df": None,
        "symbol": None,
        "conversations": {i: [] for i in range(1, 6)},
        "user_answers": {i: "" for i in range(1, 6)},
        "step_completed": {i: False for i in range(1, 6)},
        "step_summaries": {},
        "reading_profile": {},
        "round_count": {i: 0 for i in range(1, 6)},
        "total_practices": 0,
        "step_times": {i: [] for i in range(1, 6)},
        "current_step_start_time": None,
        "practice_start_time": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def reset_step_progress():
    st.session_state.step = 1
    st.session_state.conversations = {i: [] for i in range(1, 6)}
    st.session_state.user_answers = {i: "" for i in range(1, 6)}
    st.session_state.step_completed = {i: False for i in range(1, 6)}
    st.session_state.step_summaries = {}
    st.session_state.round_count = {i: 0 for i in range(1, 6)}
    st.session_state.current_step_start_time = time.time()


def load_new_symbol(code, period_value):
    with st.spinner(f"加载 {code} {period_value}分钟..."):
        df = load_data(f"{code}0", period=period_value)
        if df is None or len(df) < 100:
            st.error(f"{code} 数据加载失败")
            return False
        bar = random.randint(80, len(df) - 20)
        st.session_state.df = df
        st.session_state.symbol = code
        st.session_state.current_bar = bar
        reset_step_progress()
        st.session_state.practice_start_time = time.time()
        st.session_state.current_step_start_time = time.time()
        return True


def get_step_avg_time(step_num):
    times = st.session_state.step_times[step_num]
    if not times:
        return None
    return sum(times) / len(times)


def get_step_trend(step_num):
    times = st.session_state.step_times[step_num]
    if len(times) < 6:
        return None
    recent = sum(times[-5:]) / 5
    older = sum(times[:-5]) / len(times[:-5])
    if recent < older * 0.9:
        return "↑"
    elif recent > older * 1.1:
        return "↓"
    return "→"


# ==================== 主界面 ====================
def main():
    # 页面配置
    st.set_page_config(page_title="Al Brooks 5步训练器", layout="wide")
    
    # 自定义CSS - 柔和配色
    st.markdown("""
    <style>
    /* 主背景 */
    .stApp {
        background-color: #f5f7fa;
    }
    
    /* 侧边栏 - 柔和深色 */
    [data-testid="stSidebar"] {
        background-color: #1e293b;
    }
    
    [data-testid="stSidebar"] .stMarkdown,
    [data-testid="stSidebar"] .stMarkdown p,
    [data-testid="stSidebar"] .stMetric label,
    [data-testid="stSidebar"] .stMetric value {
        color: #e2e8f0 !important;
    }
    
    [data-testid="stSidebar"] .stMetric {
        background-color: #334155;
        border-radius: 8px;
        padding: 8px;
    }
    
    [data-testid="stSidebar"] .stProgress > div > div {
        background-color: #4f8ef7;
    }
    
    /* 侧边栏标题 */
    [data-testid="stSidebar"] h1, 
    [data-testid="stSidebar"] h2, 
    [data-testid="stSidebar"] h3,
    [data-testid="stSidebar"] .stMarkdown h3 {
        color: #ffffff !important;
    }
    
    /* 侧边栏展开器 */
    [data-testid="stSidebar"] details {
        background-color: #334155;
        border-radius: 8px;
        padding: 4px 8px;
        margin: 4px 0;
    }
    
    [data-testid="stSidebar"] summary {
        color: #cbd5e1 !important;
    }
    
    /* 侧边栏按钮 */
    [data-testid="stSidebar"] .stButton button {
        background-color: #4f8ef7;
        color: white;
        border-radius: 6px;
        border: none;
        font-size: 12px;
    }
    
    [data-testid="stSidebar"] .stButton button:hover {
        background-color: #3b7ae9;
    }
    
    /* 侧边栏滑块 */
    [data-testid="stSidebar"] .stSlider {
        color: #cbd5e1;
    }
    
    /* 主区域标题 */
    .main-title {
        color: #1e293b;
        font-size: 24px;
        font-weight: 600;
        margin-bottom: 16px;
    }
    
    /* 步骤参考区域 */
    .stExpander {
        background-color: #ffffff;
        border-radius: 8px;
        border: 1px solid #e2e8f0;
    }
    
    /* 聊天区域 */
    .stChatMessage {
        background-color: #ffffff;
        border-radius: 12px;
        padding: 8px 12px;
        margin: 4px 0;
    }
    
    /* 成功提示 */
    .stSuccess {
        background-color: #dcfce7;
        color: #166534;
    }
    
    /* 信息提示 */
    .stInfo {
        background-color: #eff6ff;
        color: #1e40af;
    }
    
    /* 速度标签 */
    .speed-up {
        color: #22c55e;
    }
    .speed-down {
        color: #ef4444;
    }
    .speed-steady {
        color: #f59e0b;
    }
    </style>
    """, unsafe_allow_html=True)

    init_state()

    if st.session_state.practice_start_time is None and st.session_state.df is not None:
        st.session_state.practice_start_time = time.time()
        st.session_state.current_step_start_time = time.time()

    # 侧边栏
    with st.sidebar:
        st.markdown("### 📊 训练统计")
        st.markdown("---")
        
        total_practices = st.session_state.total_practices
        st.metric("完成复盘次数", total_practices)
        
        st.markdown("---")
        st.markdown("**⚡ 识别速度**")
        st.markdown("*越练越快*")
        
        for i in range(1, 6):
            avg_time = get_step_avg_time(i)
            trend = get_step_trend(i)
            trend_symbol = ""
            if trend == "↑":
                trend_symbol = " ↑"
            elif trend == "↓":
                trend_symbol = " ↓"
            else:
                trend_symbol = " →"
            
            if avg_time:
                st.markdown(f"**{STEPS[i]['name']}**: {avg_time:.0f}秒{trend_symbol}")
            else:
                st.markdown(f"**{STEPS[i]['name']}**: --秒")
        
        st.markdown("---")
        st.markdown("**🧠 读盘画像**")
        if st.session_state.reading_profile:
            for step_key, weaknesses in st.session_state.reading_profile.items():
                for w, count in weaknesses.items():
                    if count >= 2:
                        st.markdown(f"⚠️ {STEPS[step_key]['name']}: {w[:12]}… ×{count}")
        else:
            st.caption("暂无数据")
            
        st.markdown("---")
        st.markdown("**🎯 今日目标**")
        st.markdown("- 完成 3 次复盘")
        st.markdown("- 每步 < 60秒")
        
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
            df = st.session_state.df
            max_bar = len(df) - 1
            new_bar = st.slider(
                "📌 K线位置", min_value=60, max_value=max_bar,
                value=st.session_state.current_bar, key="bar_slider"
            )
            if new_bar != st.session_state.current_bar:
                st.session_state.current_bar = new_bar
                reset_step_progress()
                st.session_state.practice_start_time = time.time()
                st.session_state.current_step_start_time = time.time()
                st.rerun()

            col1, col2 = st.columns(2)
            if col1.button("🎲 随机", use_container_width=True):
                new_bar = random.randint(80, max_bar - 20)
                st.session_state.current_bar = new_bar
                reset_step_progress()
                st.session_state.practice_start_time = time.time()
                st.session_state.current_step_start_time = time.time()
                st.rerun()

            if col2.button("🔄 重置", use_container_width=True):
                reset_step_progress()
                st.session_state.current_step_start_time = time.time()
                st.rerun()

            st.caption(f"当前K线: K{st.session_state.current_bar} / 共{len(df)}根")

    # 主界面
    if st.session_state.df is None:
        st.markdown('<div class="main-title">👈 请从左侧选择品种开始训练</div>', unsafe_allow_html=True)
        st.markdown("""
        <div style="background-color: #ffffff; padding: 20px; border-radius: 12px; border: 1px solid #e2e8f0;">
        <h3 style="color: #1e293b;">Al Brooks 5步框架（习惯培养版）</h3>
        <p style="color: #475569;">这不是考试，这是练习。目标是<strong>每天按固定流程复盘</strong>，形成习惯，提高识别速度。</p>
        
        <h4 style="color: #334155;">训练流程：</h4>
        <ol style="color: #475569;">
            <li>选择品种和周期</li>
            <li>按照5步框架逐步分析图表</li>
            <li>每步有2轮对话，完成后自动进入下一步</li>
            <li>完成5步后，系统记录耗时和观察完整度</li>
        </ol>
        
        <h4 style="color: #334155;">Al Brooks 5步框架：</h4>
        <ul style="color: #475569;">
            <li><strong>第1步</strong>：画线 - 找到通道、楔形、超出(Overshoot)</li>
            <li><strong>第2步</strong>：形态 - 识别双顶/双底、三角形等（形态≠预测）</li>
            <li><strong>第3步</strong>：特殊K线 - 找到IB、OB、大K线、连续强势K线</li>
            <li><strong>第4步</strong>：值得做吗 - 先评估风险回报，再决定入场</li>
            <li><strong>第5步</strong>：管理仓位 - 持仓期间的计划</li>
        </ul>
        
        <p style="color: #64748b; font-style: italic;">💡 画线没有唯一正确答案，灵活比完美更重要。</p>
        </div>
        """, unsafe_allow_html=True)
        return

    df = st.session_state.df
    bar = st.session_state.current_bar
    current_step_num = st.session_state.step
    current_step = STEPS[current_step_num]

    symbol_name = SYMBOL_NAMES.get(st.session_state.symbol, st.session_state.symbol)
    col1, col2, col3 = st.columns([2, 2, 3])
    col1.markdown(f"**{symbol_name}** ({st.session_state.symbol}) | K{bar}")
    col2.markdown(f"当前: **{current_step['name']}**")
    
    completed_count = sum(st.session_state.step_completed.values())
    col3.progress(completed_count / 5, text=f"{completed_count}/5步完成")

    fig = build_chart(df, bar, step_name=current_step["name"])
    st.plotly_chart(fig, use_container_width=True)

    # 全部完成
    if all(st.session_state.step_completed.values()):
        st.success("🎉 恭喜！完成一次完整复盘！")
        
        if st.session_state.practice_start_time:
            elapsed = time.time() - st.session_state.practice_start_time
            st.metric("本次复盘总耗时", f"{elapsed:.0f}秒")
        
        st.info("💡 复盘次数越多，识别速度会越快。继续练习！")
        
        if st.button("🔄 开始下一次复盘", type="primary"):
            new_bar = random.randint(80, len(df) - 20)
            st.session_state.current_bar = new_bar
            reset_step_progress()
            st.session_state.practice_start_time = time.time()
            st.session_state.current_step_start_time = time.time()
            st.session_state.total_practices += 1
            st.rerun()
        return

    # 当前步骤
    st.markdown("---")
    st.markdown(f"## {current_step['name']} — {current_step['title']}")
    st.markdown(f"*{current_step['question']}*")

    with st.expander("📖 本步骤参考", expanded=False):
        st.markdown("**核心观察要素：**")
        for ce in current_step["core_elements"]:
            st.markdown(f"- {ce}")
        st.success(f"✅ **好的示例：** {current_step['example_good']}")
        st.warning(f"⚠️ **不好的示例：** {current_step['example_bad']}")

    # 对话历史
    conv = st.session_state.conversations[current_step_num]
    for msg in conv:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # 当前步骤已完成
    if st.session_state.step_completed[current_step_num]:
        st.success(f"✅ {current_step['name']} 完成！")
        if current_step_num < 5:
            if st.button(f"➡️ 进入第{current_step_num + 1}步", type="primary"):
                if st.session_state.current_step_start_time:
                    elapsed = time.time() - st.session_state.current_step_start_time
                    st.session_state.step_times[current_step_num].append(elapsed)
                
                st.session_state.step = current_step_num + 1
                st.session_state.current_step_start_time = time.time()
                st.rerun()
        return

    # 用户输入
    round_num = st.session_state.round_count[current_step_num]
    
    if round_num < MAX_ROUNDS_PER_STEP:
        placeholder = "请描述你的观察..." if round_num == 0 else f"第{round_num + 1}轮，可以补充..."
        user_input = st.chat_input(placeholder)

        if user_input:
            conv.append({"role": "user", "content": user_input})
            st.session_state.user_answers[current_step_num] += f"\n[第{round_num + 1}轮] {user_input}"
            st.session_state.round_count[current_step_num] += 1

            with st.spinner("教练思考中..."):
                coach_reply = call_coach(
                    step_num=current_step_num,
                    conversation_history=conv,
                    step_summaries=st.session_state.step_summaries,
                    reading_profile=st.session_state.reading_profile,
                )

            conv.append({"role": "assistant", "content": coach_reply})
            
            if "[NEXT]" in coach_reply or round_num + 1 >= MAX_ROUNDS_PER_STEP:
                st.session_state.step_completed[current_step_num] = True
                summary = summarize_step(current_step_num, st.session_state.user_answers[current_step_num])
                st.session_state.step_summaries[current_step_num] = summary
                
            st.rerun()


if __name__ == "__main__":
    main()
