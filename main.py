"""
Al Brooks 日内机会寻找训练器 V3.0
基于V2.2的彻底重构版本

核心改变 (根据用户反馈):
1. 从“通过/失败”考试模式 -> “重复练习/形成习惯”培养模式
2. 扩充实步骤的核心要素，完全对齐Brooks视频原意
3. AI角色从“考官” -> “Brooks本人”，鼓励灵活标图，不追求完美
4. 评分逻辑从“对答案” -> 评估观察的“完整度”和“合理性”
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import akshare as ak
from openai import OpenAI
import random
import json
import re
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

# ==================== 5步流程定义 (V3.0 大幅扩展) ====================
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
        "example_bad": "有双底，要涨了。",
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
        "name": "第4步：入场点",
        "title": "标记潜在入场位置",
        "question": "你的理想入场点在哪里？为什么在这里？",
        "core_elements": [
            "入场K线编号",
            "方向：做多还是做空？",
            "入场价格类型：限价单(Limit) 还是 突破单(Stop)？",
            "具体位置：K线上方/下方多少tick？",
            "入场理由：结合第1-3步中的至少两个依据",
            "如果未触发，怎么办？",
        ],
        "example_good": "我计划在K28（看涨Pin Bar，低点在双底右底处）上方1 tick做多，使用Stop单。理由：①双底右底支撑；②下降通道出现Overshoot后收回；③Pin Bar强势收盘。如果K29收盘价低于K28低点，则取消入场。",
        "example_bad": "在K28做多，感觉要涨。",
    },
    5: {
        "name": "第5步：交易管理",
        "title": "完整的止损、目标、仓位和退出计划",
        "question": "你将如何管理这笔交易？",
        "core_elements": [
            "止损位置：具体价格（通常设在信号K线高/低点外侧）",
            "第一目标位置（至少是风险的1倍，最好是2倍以上）",
            "风险回报比(R:R)至少要达到2:1",
            "仓位规模：根据你的风险和账户资金决定",
            "实现概率：这笔交易成功的概率有多高？(高/中/低)",
            "提前退出条件：什么情况下会提前离场？",
        ],
        "example_good": "做多止损设在K28低点下方1 tick（3540），入场价3560，风险20点。第一目标3600（1R），第二目标3640（2R，前高附近）。仓位规模为总资金的2%。这笔交易成功概率高，因为它结合了形态、K线和通道的多重信号。若入场后价格在K28收盘价下方持续3根K线，则提前离场。",
        "example_bad": "止损在低点，目标在高点，风险回报2:1。",
    },
}

# ==================== 结构分析函数 ====================
def find_swing_points(df, lookback=60, pivot_window=3):
    """识别摆动高低点（swing high/low）"""
    start = max(0, len(df) - lookback)
    sub = df.iloc[start:].copy().reset_index(drop=True)
    n = len(sub)
    w = pivot_window
    swing_highs, swing_lows = [], []

    for i in range(w, n - w):
        h = sub.iloc[i]["high"]
        l = sub.iloc[i]["low"]
        orig_idx = start + i

        if all(sub.iloc[i - j]["high"] < h for j in range(1, w + 1)) and \
           all(sub.iloc[i + j]["high"] < h for j in range(1, w + 1)):
            swing_highs.append({"idx": orig_idx, "price": h})

        if all(sub.iloc[i - j]["low"] > l for j inrange(1, w + 1)) and \
           all(sub.iloc[i + j]["low"] > l for j in range(1, w + 1)):
            swing_lows.append({"idx": orig_idx, "price": l})

    return swing_highs, swing_lows


def detect_trend(swing_highs, swing_lows):
    if len(swing_highs) < 3 or len(swing_lows) < 3:
        return "数据不足（需要至少3个摆动高点+3个摆动低点）"

    h = swing_highs[-3:]
    l = swing_lows[-3:]

    hh_count = (1 if h[1]["price"] > h[0]["price"] else 0) + (1 if h[2]["price"] > h[1]["price"] else 0)
    hl_count = (1 if l[1]["price"] > l[0]["price"] else 0) + (1 if l[2]["price"] > l[1]["price"] else 0)
    lh_count = (1 if h[1]["price"] < h[0]["price"] else 0) + (1 if h[2]["price"] < h[1]["price"] else 0)
    ll_count = (1 if l[1]["price"] < l[0]["price"] else 0) + (1 if l[2]["price"] < l[1]["price"] else 0)

    if hh_count >= 2 and hl_count >= 2:
        return "上升趋势"
    elif lh_count >= 2 and ll_count >= 2:
        return "下降趋势"
    else:
        return "横盘震荡"


def detect_bar_patterns(df, bar, lookback=40):
    """识别K线形态：V3.0 扩展版本"""
    start = max(0, bar - lookback)
    sub = df.iloc[start:bar + 1].copy().reset_index(drop=True)
    n = len(sub)
    patterns = []

    for i in range(1, n):
        prev = sub.iloc[i - 1]
        curr = sub.iloc[i]
        orig_idx = start + i
        ph, pl = prev["high"], prev["low"]
        ch, cl = curr["high"], curr["low"]
        co, cc = curr["open"], curr["close"]
        body = abs(cc - co)
        total = ch - cl
        body_ratio = body / total if total > 0 else 0

        # 内包线
        if ch < ph and cl > pl:
            patterns.append(f"K{orig_idx}: 内包线(IB)")
        # 外包线
        elif ch > ph and cl < pl:
            patterns.append(f"K{orig_idx}: 外包线(OB)")

        # 大K线（实体占比>80%）
        if body_ratio > 0.80:
            direction = "阳线" if cc > co else "阴线"
            if (cc > co and cc > ch - total * 0.2) or (cc < co and cc < cl + total * 0.2):
                close_pos = "收盘在端点"
            else:
                close_pos = "收盘在中部"
            patterns.append(f"K{orig_idx}: 大{direction}({body_ratio*100:.0f}%实体, {close_pos})")

    # 连续同向强势K线（无重叠）
    streak = 1
    streak_dir = None
    low_wick_ratio = 1.0
    for i in range(n - 2, -1, -1):
        row = sub.iloc[i]
        curr = sub.iloc[i+1]
        is_bull = row["close"] >= row["open"]
        if streak_dir is None:
            streak_dir = is_bull
        elif is_bull == streak_dir:
            # 检查与前一根K线是否有重叠
            overlap = min(curr["high"], row["high"]) - max(curr["low"], row["low"])
            if overlap <= 0:
                streak += 1
            else:
                break
        else:
            break
    if streak >= 3:
        direction = "多头" if streak_dir else "空头"
        patterns.append(f"最近{streak}根连续强势{direction}K线（无明显重叠，截至K{bar}）")

    # IOI模式
    for i in range(2, n):
        k1 = sub.iloc[i-2]
        k2 = sub.iloc[i-1]
        k3 = sub.iloc[i]
        if (k2["high"] < k1["high"] and k2["low"] > k1["low"]) and \
           (k3["high"] > k2["high"] and k3["low"] < k2["low"]):
            patterns.append(f"K{start+i-2}-K{start+i}: IOI模式 (外包->内包->外包)")

    return patterns[-20:]


def detect_double_top_bottom(swing_highs, swing_lows):
    results = []
    if len(swing_highs) >= 2:
        h1, h2 = swing_highs[-2], swing_highs[-1]
        diff_pct = abs(h2["price"] - h1["price"]) / h1["price"] if h1["price"] > 0 else 1
        if diff_pct < 0.015:  # 容忍度 1.5%
            if h2["price"] > h1["price"]:
                label = "更高高点"
            elif h2["price"] < h1["price"]:
                label = "更低高点"
            else:
                label = "等高"
            results.append(f"{label}双顶：K{h1['idx']}({h1['price']:.0f}) → K{h2['idx']}({h2['price']:.0f})")

    if len(swing_lows) >= 2:
        l1, l2 = swing_lows[-2], swing_lows[-1]
        diff_pct = abs(l2["price"] - l1["price"]) / l1["price"] if l1["price"] > 0 else 1
        if diff_pct < 0.015:
            if l2["price"] > l1["price"]:
                label = "更高低点"
            elif l2["price"] < l1["price"]:
                label = "更低低点"
            else:
                label = "等低"
            results.append(f"{label}双底：K{l1['idx']}({l1['price']:.0f}) → K{l2['idx']}({l2['price']:.0f})")

    return results


def build_structure_report(df, bar):
    """生成完整结构分析报告，供AI使用"""
    swing_highs, swing_lows = find_swing_points(df, lookback=80)
    swing_highs = [s for s in swing_highs if s["idx"] <= bar]
    swing_lows = [s for s in swing_lows if s["idx"] <= bar]

    trend = detect_trend(swing_highs, swing_lows)
    bar_patterns = detect_bar_patterns(df, bar, lookback=50)
    double_patterns = detect_double_top_bottom(swing_highs, swing_lows)

    start = max(0, bar - 60)
    sub = df.iloc[start:bar + 1]
    price_range_high = float(sub["high"].max())
    price_range_low = float(sub["low"].min())
    current = df.iloc[bar]

    lines = [
        "═══════ 结构分析报告 ═══════",
        f"当前K线: K{bar} | O={float(current['open']):.0f} H={float(current['high']):.0f} L={float(current['low']):.0f} C={float(current['close']):.0f}",
        f"60根K线区间: {price_range_low:.0f} ~ {price_range_high:.0f}",
        "",
        "【趋势判断】",
        trend,
        "",
        "【摆动高点】（最近5个）",
    ]
    for sh in swing_highs[-5:]:
        lines.append(f"  K{sh['idx']}: {sh['price']:.0f}")
    lines.append("【摆动低点】（最近5个）")
    for sl in swing_lows[-5:]:
        lines.append(f"  K{sl['idx']}: {sl['price']:.0f}")
    lines += ["", "【双顶/双底检测】"]
    lines += [f"  {dp}" for dp in double_patterns] if double_patterns else ["  未检测到明显双顶或双底"]
    lines += ["", "【K线形态识别】（最近50根）"]
    lines += [f"  {bp}" for bp in bar_patterns] if bar_patterns else ["  未发现特殊K线形态"]

    return "\n".join(lines)


# ==================== AI 提示词 (V3.0 彻底重写) ====================
COACH_SYSTEM = """你是 Al Brooks 价格行为分析教练。

【你的角色】
- 你正在训练一个交易员养成“每天按固定流程复盘”的习惯。
- 你不是考官，你是陪练。
- 目标不是让用户“答对”，而是帮他“逐步形成自己的观察框架”。

【如何引导】
- 优先观察用户是否遗漏了当前步骤最核心的几个要素。
- 鼓励他引用具体K线编号、价格、形态。
- 如果他提出的画法/判断与后台分析不一致，只要逻辑合理、有理有据，都视为有效。
- 如果观察明显错误，用提问引导他重新看，而不是直接否定。
- 对他“已经做得好的部分”要给予正反馈。

【灵活性原则】
- 双顶/双底很少完美，不要求精准。
- 通道画法可以有多种，关键是能否解释得通。
- 灵活判断比追求完美更重要。

【输出要求】
- 反馈请控制在120字以内。
- 不要输出PASS/FAIL。
- 如果你认为他已经描述得很完整，并且给出了具体K线编号和理由，可以说“这一轮的观察很到位，可以进入下一步了”。
- 如果他遗漏较多，可以说“这次我们主要看看[具体缺失的点]，你注意到了吗？”

【当前训练步骤】
{step_info}

【用户历史薄弱点】
{profile_text}

【图表结构数据】（仅供参考，不要直接说出“后台检测到…”）
{structure_report}

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


# ==================== AI 调用 ====================
def get_ai_client():
    api_key = st.secrets.get("OPENAI_API_KEY", "")
    base_url = st.secrets.get("OPENAI_BASE_URL", "https://api.deepseek.com")
    model = st.secrets.get("OPENAI_MODEL", "deepseek-chat")
    return OpenAI(base_url=base_url, api_key=api_key), model


def call_coach(step_num, conversation_history, structure_report, step_summaries, reading_profile):
    client, model = get_ai_client()
    step = STEPS[step_num]
    system = COACH_SYSTEM.format(
        step_info=build_step_info(step_num),
        profile_text=build_profile_text(reading_profile),
        structure_report=structure_report,
        previous_findings=build_previous_findings(step_summaries),
    )
    messages = [{"role": "system", "content": system}] + conversation_history
    try:
        resp = client.chat.completions.create(
            model=model, messages=messages, temperature=0.3, max_tokens=400
        )
        return resp.choices[0].message.content
    except Exception as e:
        return f"API错误: {e}"


def summarize_step(step_num, user_answers_text):
    client, model = get_ai_client()
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "用30字以内总结用户在本步骤的关键发现，只输出发现内容。"},
                {"role": "user", "content": user_answers_text}
            ],
            temperature=0.0,
            max_tokens=100,
        )
        return resp.choices[0].message.content.strip()
    except:
        return user_answers_text[:100]


def update_reading_profile(step_num, missing_elements):
    if not missing_elements:
        return
    if step_num not in st.session_state.reading_profile:
        st.session_state.reading_profile[step_num] = {}
    for m in missing_elements:
        st.session_state.reading_profile[step_num][m] = \
            st.session_state.reading_profile[step_num].get(m, 0) + 1


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
    except:
        return None


# ==================== 图表绘制 ====================
def build_chart(df, bar, swing_highs=None, swing_lows=None, step_name=""):
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
                font=dict(size=7, color="#999999"),
                yshift=-14, row=1, col=1
            )

    bar_pos = original_indices.index(bar) if bar in original_indices else len(original_indices) - 1
    fig.add_vline(x=bar_pos, line_dash="dash", line_color="#ff9800", line_width=1.5, opacity=0.8)

    if swing_highs:
        for sh in swing_highs:
            if start <= sh["idx"] <= bar:
                pos = sh["idx"] - start
                fig.add_annotation(
                    x=pos, y=sh["price"], text="H", showarrow=False,
                    font=dict(size=9, color="#ff6b6b"), yshift=8, row=1, col=1
                )
    if swing_lows:
        for sl in swing_lows:
            if start <= sl["idx"] <= bar:
                pos = sl["idx"] - start
                fig.add_annotation(
                    x=pos, y=sl["price"], text="L", showarrow=False,
                    font=dict(size=9, color="#4ecdc4"), yshift=-10, row=1, col=1
                )

    if step_name:
        fig.add_annotation(
            x=0.02, y=0.98, xref="paper", yref="paper",
            text=f"📌 {step_name}", showarrow=False,
            font=dict(size=12, color="#e0e0e0"),
            bgcolor="rgba(0,0,0,0.5)", borderpad=4
        )

    fig.update_layout(
        xaxis_rangeslider_visible=False,
        height=480,
        margin=dict(l=5, r=5, t=5, b=5),
        paper_bgcolor="#1a1a2e",
        plot_bgcolor="#16213e",
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
        "structure_report": None,
        "swing_highs": [],
        "swing_lows": [],
        "conversations": {i: [] for i in range(1, 6)},
        "user_answers": {i: "" for i in range(1, 6)},
        "step_passed": {i: False for i in range(1, 6)},
        "step_summaries": {},
        "reading_profile": {},
        "round_count": {i: 0 for i in range(1, 6)},
        # V3.0 新增：统计信息，用于“习惯培养”模式
        "total_practices": 0,  # 完成的完整复盘次数
        "step_scores": {i: [] for i in range(1, 6)},  # 存储每次练习每步的分数
        "practice_times": [],  # 存储每次练习的耗时（秒）
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def reset_step_progress():
    st.session_state.step = 1
    st.session_state.conversations = {i: [] for i in range(1, 6)}
    st.session_state.user_answers = {i: "" for i in range(1, 6)}
    st.session_state.step_passed = {i: False for i in range(1, 6)}
    st.session_state.step_summaries = {}
    st.session_state.round_count = {i: 0 for i in range(1, 6)}


def load_new_symbol(code, period_value):
    with st.spinner(f"加载 {code} {period_value}分钟..."):
        df = load_data(f"{code}0", period=period_value)
        if df is None or len(df) < 100:
            st.error(f"{code} 数据加载失败")
            return False
        bar = random.randint(80, len(df) - 20)
        sh, sl = find_swing_points(df, lookback=80)
        st.session_state.df = df
        st.session_state.symbol = code
        st.session_state.current_bar = bar
        st.session_state.swing_highs = sh
        st.session_state.swing_lows = sl
        st.session_state.structure_report = build_structure_report(df, bar)
        reset_step_progress()
        return True


# ==================== 模拟评分函数（用于统计）====================
def simulate_score(user_answer, core_elements):
    """
    模拟一个简单的评分：根据用户回答中出现的核心要素数量计算完整度。
    真实场景下可以调用AI或更复杂的逻辑，这里用于演示统计面板。
    """
    if not user_answer:
        return 0
    score = 0
    for element in core_elements:
        # 简单检查核心词是否出现，真实场景可以更精细
        if any(keyword in user_answer for keyword in element.split()[:2]): # 非常粗糙的模拟
            score += 1
    return (score / len(core_elements)) * 100


# ==================== 主界面 ====================
def main():
    st.set_page_config(page_title="Al Brooks 5步训练器 (习惯培养版)", layout="wide")

    st.markdown("""
    <style>
    .main { background-color: #0f0f1a; }
    .stSidebar { background-color: #1a1a2e; }
    .stProgress > div > div { background-color: #4ecdc4; }
    .stButton button { background-color: #4f8ef7; color: white; border-radius: 4px; border: none; }
    .stButton button:hover { background-color: #6ca3ff; }
    </style>
    """, unsafe_allow_html=True)

    init_state()

    # ── 侧边栏（统计信息）──────────────────────
    with st.sidebar:
        st.markdown("### 📊 训练统计")
        st.markdown("---")
        
        # 显示完成次数
        total_practices = st.session_state.total_practices
        st.metric("完成复盘次数", total_practices)
        
        # 显示平均分
        all_scores = []
        for step_scores in st.session_state.step_scores.values():
            all_scores.extend(step_scores)
        avg_score = sum(all_scores) / len(all_scores) if all_scores else 0
        st.metric("平均观察完整度", f"{avg_score:.1f}%")
        
        # 显示最近一次耗时
        if st.session_state.practice_times:
            last_time = st.session_state.practice_times[-1]
            st.metric("最近一次复盘耗时", f"{last_time:.0f}秒")
            if len(st.session_state.practice_times) >= 2:
                improvement = st.session_state.practice_times[-2] - last_time
                st.caption(f"比上次 {'快' if improvement > 0 else '慢'} {abs(improvement):.0f}秒")
        
        st.markdown("---")
        st.markdown("**📈 各步平均分**")
        for i in range(1, 6):
            scores = st.session_state.step_scores[i]
            avg = sum(scores) / len(scores) if scores else 0
            st.progress(avg/100, text=f"步骤{i}: {avg:.0f}分")
        
        st.markdown("---")
        st.markdown("**🧠 读盘画像**")
        if st.session_state.reading_profile:
            for step_key, weaknesses in st.session_state.reading_profile.items():
                for w, count in weaknesses.items():
                    if count >= 2:
                        st.markdown(f"⚠️ 步骤{step_key}: {w[:10]}… ×{count}")
        else:
            st.caption("暂无数据，完成一次复盘后显示")
            
        st.markdown("---")
        st.markdown("**🎯 今日目标**")
        st.markdown("- 完成 3 次完整复盘")
        st.markdown("- 平均分 > 75%")
        st.markdown("- 单次耗时 < 5 分钟")
        
        st.markdown("---")
        # 品种选择部分保持不变
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
                "K线位置", min_value=60, max_value=max_bar,
                value=st.session_state.current_bar, key="bar_slider"
            )
            if new_bar != st.session_state.current_bar:
                st.session_state.current_bar = new_bar
                sh, sl = find_swing_points(df, lookback=80)
                st.session_state.swing_highs = sh
                st.session_state.swing_lows = sl
                st.session_state.structure_report = build_structure_report(df, new_bar)
                reset_step_progress()
                st.rerun()

            col1, col2 = st.columns(2)
            if col1.button("🎲 随机", use_container_width=True):
                new_bar = random
