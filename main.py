"""
Al Brooks 日内机会寻找训练器 V4.4
修复：统计数据显示问题 + 侧边栏白底配色
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
import json

# ==================== 简单线性回归（纯 NumPy，无 sklearn 依赖）====================
def simple_linear_regression(x, y):
    """简单线性回归，返回斜率和 R²"""
    n = len(x)
    if n < 2:
        return 0, 0
    
    x_mean = sum(x) / n
    y_mean = sum(y) / n
    
    numerator = sum((x[i] - x_mean) * (y[i] - y_mean) for i in range(n))
    denominator = sum((x[i] - x_mean) ** 2 for i in range(n))
    
    if denominator == 0:
        return 0, 0
    
    slope = numerator / denominator
    
    ss_res = sum((y[i] - (slope * x[i] + (y_mean - slope * x_mean))) ** 2 for i in range(n))
    ss_tot = sum((y[i] - y_mean) ** 2 for i in range(n))
    
    r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
    
    return slope, r2


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

MIN_ROUNDS = 1
MAX_ROUNDS = 3

# ==================== 5步流程定义 ====================
STEPS = {
    1: {
        "name": "第1步：画线",
        "title": "寻找通道、楔形和超出(Overshoot)",
        "question": "你会如何画线？请说明你的选择和理由。",
        "core_elements": ["通道方向", "通道边界", "楔形", "超出(Overshoot)", "备选画法"],
        "example_good": "我用K5、K15、K25的高点画了一条下降趋势线。K8、K18的低点构成了平行支撑线，形成一个下降通道。K18的低点向下刺穿了支撑线形成Overshoot，但很快收回。",
        "example_bad": "这是一个下降通道。",
    },
    2: {
        "name": "第2步：形态",
        "title": "识别双顶/双底、三角形等反转形态",
        "question": "图表上出现了哪些反转或延续形态？",
        "core_elements": ["双顶或双底", "收缩三角形", "扩展三角形", "推进波", "失败突破"],
        "example_good": "K12和K28构成更高低点的双底。K15-K20之间是收缩三角形。K22向上假突破后回落。",
        "example_bad": "有双底，要涨了。",
    },
    3: {
        "name": "第3步：特殊K线",
        "title": "识别信号K线",
        "question": "有哪些特殊的单根K线或K线组合？",
        "core_elements": ["大K线", "惊讶K线", "内包线(IB)", "外包线(OB)", "IOI模式", "连续强势K线"],
        "example_good": "K15是大阳线，实体92%，高位收盘。K16是内包线。K17是外包线，形成IOI。K22是惊讶K线。",
        "example_bad": "K15是大K线。",
    },
    4: {
        "name": "第4步：值得做吗",
        "title": "评估概率、空间和风险",
        "question": "这笔交易值得做吗？",
        "core_elements": ["方向", "成功概率", "空间至少2R", "失败代价", "是否值得"],
        "example_good": "空头机会。双顶+下降通道，概率高。风险20点，目标40点以上。值得做。",
        "example_bad": "感觉要涨。",
    },
    5: {
        "name": "第5步：管理仓位",
        "title": "持仓期间的管理计划",
        "question": "入场后如何管理？",
        "core_elements": ["仓位规模", "提前退出条件", "移动止损", "分批止盈"],
        "example_good": "2%仓位。若3根K线反向则离场。涨到1R后移动止损保本。",
        "example_bad": "拿着等涨。",
    },
}

# ==================== 增强的结构分析类 ====================
class StructureAnalyzer:
    def __init__(self, df, bar, lookback=80):
        self.df = df
        self.bar = bar
        self.start = max(0, bar - lookback)
        self.sub = df.iloc[self.start:bar+1].copy().reset_index(drop=True)
        self.original_indices = list(range(self.start, bar+1))
        self.swing_highs = []
        self.swing_lows = []
        self._find_swing_points()
    
    def _find_swing_points(self, pivot_window=2):
        n = len(self.sub)
        w = pivot_window
        for i in range(w, n - w):
            h = self.sub.iloc[i]["high"]
            l = self.sub.iloc[i]["low"]
            orig_idx = self.original_indices[i]
            
            left_higher = all(self.sub.iloc[i - j]["high"] < h for j in range(1, w+1))
            right_higher = all(self.sub.iloc[i + j]["high"] < h for j in range(1, w+1))
            if left_higher and right_higher:
                self.swing_highs.append({"idx": orig_idx, "price": h})
            
            left_lower = all(self.sub.iloc[i - j]["low"] > l for j in range(1, w+1))
            right_lower = all(self.sub.iloc[i + j]["low"] > l for j in range(1, w+1))
            if left_lower and right_lower:
                self.swing_lows.append({"idx": orig_idx, "price": l})
    
    def detect_channel(self):
        if len(self.swing_highs) < 2 or len(self.swing_lows) < 2:
            return "数据不足"
        highs = self.swing_highs[-3:]
        lows = self.swing_lows[-3:]
        if len(highs) >= 2:
            high_slope = (highs[-1]["price"] - highs[-2]["price"]) / max(1, highs[-1]["idx"] - highs[-2]["idx"])
        else:
            high_slope = 0
        if len(lows) >= 2:
            low_slope = (lows[-1]["price"] - lows[-2]["price"]) / max(1, lows[-1]["idx"] - lows[-2]["idx"])
        else:
            low_slope = 0
        avg_slope = (high_slope + low_slope) / 2
        if avg_slope > 0.1:
            return "上升通道"
        elif avg_slope < -0.1:
            return "下降通道"
        else:
            return "水平通道（震荡）"
    
    def detect_wedge(self):
        if len(self.swing_highs) >= 3:
            x = [h["idx"] for h in self.swing_highs[-3:]]
            y = [h["price"] for h in self.swing_highs[-3:]]
            slope, r2 = simple_linear_regression(x, y)
            if r2 > 0.9:
                return True
        if len(self.swing_lows) >= 3:
            x = [l["idx"] for l in self.swing_lows[-3:]]
            y = [l["price"] for l in self.swing_lows[-3:]]
            slope, r2 = simple_linear_regression(x, y)
            if r2 > 0.9:
                return True
        return False
    
    def detect_double_top_bottom(self):
        results = []
        if len(self.swing_highs) >= 2:
            h1, h2 = self.swing_highs[-2], self.swing_highs[-1]
            diff_pct = abs(h2["price"] - h1["price"]) / h1["price"] if h1["price"] > 0 else 1
            if diff_pct < 0.02:
                results.append(f"双顶" if h2["price"] <= h1["price"] else f"更高高点双顶")
        if len(self.swing_lows) >= 2:
            l1, l2 = self.swing_lows[-2], self.swing_lows[-1]
            diff_pct = abs(l2["price"] - l1["price"]) / l1["price"] if l1["price"] > 0 else 1
            if diff_pct < 0.02:
                results.append(f"双底" if l2["price"] >= l1["price"] else f"更低低点双底")
        return results
    
    def detect_triangle(self):
        if len(self.swing_highs) < 2 or len(self.swing_lows) < 2:
            return None
        high_diff = self.swing_highs[-1]["price"] - self.swing_highs[-2]["price"]
        low_diff = self.swing_lows[-1]["price"] - self.swing_lows[-2]["price"]
        if high_diff < 0 and low_diff > 0:
            return "收缩三角形"
        elif high_diff > 0 and low_diff < 0:
            return "扩展三角形"
        return None
    
    def detect_pushes(self):
        pushes = []
        prices = []
        for sh in self.swing_highs[-5:]:
            prices.append(("H", sh["price"], sh["idx"]))
        for sl in self.swing_lows[-5:]:
            prices.append(("L", sl["price"], sl["idx"]))
        prices.sort(key=lambda x: x[2])
        if len(prices) >= 4:
            push_count = 1
            for i in range(1, len(prices)):
                if prices[i][1] > prices[i-1][1] and prices[i][0] == "H":
                    push_count += 1
                elif prices[i][1] < prices[i-1][1] and prices[i][0] == "L":
                    push_count += 1
                else:
                    if push_count >= 2:
                        pushes.append(f"{push_count}推")
                    push_count = 1
            if push_count >= 2:
                pushes.append(f"{push_count}推")
        return pushes
    
    def detect_special_bars(self):
        patterns = []
        ranges = []
        for i in range(max(0, len(self.sub)-20), len(self.sub)):
            ranges.append(self.sub.iloc[i]["high"] - self.sub.iloc[i]["low"])
        avg_range = sum(ranges) / len(ranges) if ranges else 0
        
        for i in range(1, len(self.sub)):
            prev = self.sub.iloc[i-1]
            curr = self.sub.iloc[i]
            orig_idx = self.original_indices[i]
            co, cc = curr["open"], curr["close"]
            body = abs(cc - co)
            total = curr["high"] - curr["low"]
            body_ratio = body / total if total > 0 else 0
            
            if body_ratio > 0.80:
                direction = "阳线" if cc > co else "阴线"
                patterns.append({"type": "大K线", "idx": orig_idx})
            
            if total > avg_range * 1.5 and avg_range > 0:
                patterns.append({"type": "惊讶K线", "idx": orig_idx})
            
            if curr["high"] < prev["high"] and curr["low"] > prev["low"]:
                patterns.append({"type": "IB", "idx": orig_idx})
            elif curr["high"] > prev["high"] and curr["low"] < prev["low"]:
                patterns.append({"type": "OB", "idx": orig_idx})
        
        # IOI检测
        for i in range(2, len(self.sub)):
            k1, k2, k3 = self.sub.iloc[i-2], self.sub.iloc[i-1], self.sub.iloc[i]
            if (k2["high"] < k1["high"] and k2["low"] > k1["low"]) and \
               (k3["high"] > k2["high"] and k3["low"] < k2["low"]):
                patterns.append({"type": "IOI", "idx": self.original_indices[i]})
        
        return patterns
    
    def generate_full_report(self):
        channel = self.detect_channel()
        wedge = self.detect_wedge()
        double_patterns = self.detect_double_top_bottom()
        triangle = self.detect_triangle()
        pushes = self.detect_pushes()
        special_bars = self.detect_special_bars()
        
        detected_elements = []
        
        report_lines = [
            "═══════ AI独立分析结果 ═══════",
            f"分析范围: K{self.start} ~ K{self.bar}",
            "",
            "【第1步：通道和楔形】",
            f"  通道: {channel}",
            f"  楔形: {'✓' if wedge else '✗'}",
        ]
        if wedge:
            detected_elements.append("楔形")
        
        report_lines.append("")
        report_lines.append("【第2步：形态】")
        if double_patterns:
            report_lines.append(f"  双顶/双底: {', '.join(double_patterns)}")
            detected_elements.extend(double_patterns)
        else:
            report_lines.append("  双顶/双底: 无")
        
        if triangle:
            report_lines.append(f"  三角形: {triangle}")
            detected_elements.append("三角形")
        else:
            report_lines.append("  三角形: 无")
        
        if pushes:
            report_lines.append(f"  推进波: {', '.join(pushes)}")
            detected_elements.extend(pushes)
        else:
            report_lines.append("  推进波: 无")
        
        report_lines.append("")
        report_lines.append("【第3步：特殊K线】")
        bar_types = {}
        for bar in special_bars:
            t = bar["type"]
            bar_types[t] = bar_types.get(t, 0) + 1
        for t, count in bar_types.items():
            report_lines.append(f"  {t}: {count}次")
            if t not in ["高位收盘", "低位收盘"]:
                detected_elements.append(t)
        
        return "\n".join(report_lines), list(set(detected_elements))


# ==================== AI 提示词 ====================
COACH_SYSTEM = """你是 Al Brooks 价格行为分析教练。

【核心原则】
- 你已经独立完成了图表的结构分析
- 基于你的分析结果，批改用户的观察

【批改规则】
1. 用户遗漏了实际存在的形态 → 追问
2. 用户说了实际不存在的形态 → 友善指出
3. 用户判断合理但与你不同 → 认可灵活性

【反证原则】
- 追问："什么情况会证明你是错的？"

【输出格式】
- 反馈控制在120字以内
- 如果用户已覆盖主要形态，输出"[NEXT]"
- 否则输出"[CONTINUE]"

【AI独立分析结果】
{structure_report}

【实际存在的形态】
{detected_elements}

【当前训练步骤】
{step_info}

【用户历史薄弱点】
{profile_text}
"""


def build_step_info(step_num):
    step = STEPS[step_num]
    lines = [f"步骤：{step['name']}", f"问题：{step['question']}", "观察要素："]
    for ce in step["core_elements"]:
        lines.append(f"  - {ce}")
    return "\n".join(lines)


def build_profile_guidance(reading_profile):
    if not reading_profile:
        return "暂无"
    items = []
    for step_key, weaknesses in reading_profile.items():
        for w, count in weaknesses.items():
            if count >= 2:
                items.append(f"第{step_key}步常漏：{w}")
    return "\n".join(items) if items else "无明显薄弱点"


# ==================== AI 调用函数 ====================
def _gpt(messages):
    try:
        api_key = st.secrets.get("OPENAI_API_KEY", "")
        base_url = st.secrets.get("OPENAI_BASE_URL", "https://api.deepseek.com")
        model = st.secrets.get("OPENAI_MODEL", "deepseek-chat")
    except Exception:
        import os
        api_key = os.environ.get("OPENAI_API_KEY", "")
        base_url = os.environ.get("OPENAI_BASE_URL", "https://api.deepseek.com")
        model = os.environ.get("OPENAI_MODEL", "deepseek-chat")
    
    if not api_key:
        return "【提示】请配置API密钥\n[CONTINUE]"
    
    try:
        client = OpenAI(base_url=base_url, api_key=api_key)
        resp = client.chat.completions.create(
            model=model, messages=messages, temperature=0.2, max_tokens=700
        )
        return resp.choices[0].message.content
    except Exception as e:
        return f"【API错误】{str(e)}\n[CONTINUE]"


def call_coach(step_num, conversation_history, structure_report, detected_elements, 
               reading_profile, round_num, max_rounds):
    system_prompt = COACH_SYSTEM.format(
        structure_report=structure_report,
        detected_elements="、".join(detected_elements) if detected_elements else "无明显形态",
        step_info=build_step_info(step_num),
        profile_text=build_profile_guidance(reading_profile),
    )
    messages = [{"role": "system", "content": system_prompt}] + conversation_history
    response = _gpt(messages)
    
    if "[NEXT]" not in response and "[CONTINUE]" not in response:
        if round_num >= max_rounds:
            return response + "\n[NEXT]"
        return response + "\n[CONTINUE]"
    return response


def summarize_step(step_num, user_answers_text):
    messages = [
        {"role": "system", "content": "用30字以内总结用户的关键发现。"},
        {"role": "user", "content": user_answers_text}
    ]
    return _gpt(messages)


def calculate_coverage(user_answer, detected_elements):
    if not detected_elements:
        return 0, 0, 1.0
    user_lower = user_answer.lower()
    covered = 0
    for element in detected_elements:
        if element in user_answer or element.lower() in user_lower:
            covered += 1
    ratio = covered / len(detected_elements)
    return covered, len(detected_elements), ratio


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
def build_chart(df, bar, step_name=""):
    end = bar + 1
    start = max(0, end - 80)
    plot_df = df.iloc[start:end].copy().reset_index(drop=True)
    original_indices = list(range(start, end))

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        vertical_spacing=0.02, row_heights=[0.8, 0.2])

    fig.add_trace(go.Candlestick(x=plot_df.index,
        open=plot_df["open"], high=plot_df["high"],
        low=plot_df["low"], close=plot_df["close"],
        showlegend=False, increasing_line_color="#ef5350", decreasing_line_color="#26a69a"), row=1, col=1)

    vol_colors = ["#ef5350" if c >= o else "#26a69a" for o, c in zip(plot_df["open"], plot_df["close"])]
    fig.add_trace(go.Bar(x=plot_df.index, y=plot_df["volume"],
        marker_color=vol_colors, showlegend=False, opacity=0.5), row=2, col=1)

    for idx, orig_idx in enumerate(original_indices):
        if orig_idx % 5 == 0:
            row_data = plot_df.iloc[idx]
            fig.add_annotation(x=idx, y=row_data["low"], text=str(orig_idx),
                showarrow=False, font=dict(size=7, color="#888888"), yshift=-14, row=1, col=1)

    bar_pos = original_indices.index(bar) if bar in original_indices else len(original_indices) - 1
    fig.add_vline(x=bar_pos, line_dash="dash", line_color="#ff9800", line_width=1.5, opacity=0.8)

    if step_name:
        fig.add_annotation(x=0.02, y=0.98, xref="paper", yref="paper",
            text=f"📌 {step_name}", showarrow=False,
            font=dict(size=12, color="#ffffff"), bgcolor="rgba(0,0,0,0.6)", borderpad=4)

    fig.update_layout(xaxis_rangeslider_visible=False, height=480,
        margin=dict(l=5, r=5, t=5, b=5),
        paper_bgcolor="#0e1117", plot_bgcolor="#1a1a2e", font=dict(color="#e0e0e0"))
    fig.update_xaxes(showgrid=True, gridcolor="#2d3561", gridwidth=0.5)
    fig.update_yaxes(showgrid=True, gridcolor="#2d3561", gridwidth=0.5)
    return fig


# ==================== Session State ====================
def init_state():
    defaults = {
        "step": 1, "current_bar": 80, "df": None, "symbol": None,
        "structure_report": None, "detected_elements": [],
        "conversations": {i: [] for i in range(1, 6)},
        "user_answers": {i: "" for i in range(1, 6)},
        "step_completed": {i: False for i in range(1, 6)},
        "step_summaries": {}, "reading_profile": {},
        "round_count": {i: 0 for i in range(1, 6)},
        "total_practices": 0,
        "step_times": {i: [] for i in range(1, 6)},
        "step_coverage": {i: [] for i in range(1, 6)},
        "current_step_start_time": None, "practice_start_time": None,
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
        analyzer = StructureAnalyzer(df, bar, lookback=80)
        structure_report, detected_elements = analyzer.generate_full_report()
        st.session_state.df = df
        st.session_state.symbol = code
        st.session_state.current_bar = bar
        st.session_state.structure_report = structure_report
        st.session_state.detected_elements = detected_elements
        reset_step_progress()
        st.session_state.practice_start_time = time.time()
        st.session_state.current_step_start_time = time.time()
        return True


# ==================== 主界面 ====================
def main():
    st.set_page_config(page_title="Al Brooks 5步训练器", layout="wide")
    
    # 白底配色 CSS
    st.markdown("""
    <style>
    .stApp { background-color: #ffffff; }
    
    /* 侧边栏 - 白底深色字 */
    [data-testid="stSidebar"] {
        background-color: #f8f9fa !important;
        border-right: 1px solid #e9ecef;
    }
    [data-testid="stSidebar"] .stMarkdown,
    [data-testid="stSidebar"] .stMarkdown p,
    [data-testid="stSidebar"] .stMetric label,
    [data-testid="stSidebar"] .stMetric value {
        color: #212529 !important;
    }
    [data-testid="stSidebar"] .stMetric {
        background-color: #e9ecef;
        border-radius: 8px;
        padding: 8px;
    }
    [data-testid="stSidebar"] .stButton button {
        background-color: #e9ecef;
        color: #212529;
        border-radius: 6px;
        border: 1px solid #dee2e6;
    }
    [data-testid="stSidebar"] .stButton button:hover {
        background-color: #dee2e6;
    }
    [data-testid="stSidebar"] details {
        background-color: #e9ecef;
        border-radius: 8px;
        padding: 4px 8px;
    }
    [data-testid="stSidebar"] summary {
        color: #212529 !important;
    }
    [data-testid="stSidebar"] .stSlider label {
        color: #212529 !important;
    }
    
    /* 主区域 */
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
    .stSuccess {
        background-color: #d1e7dd;
        color: #0f5132;
    }
    .stInfo {
        background-color: #cfe2ff;
        color: #084298;
    }
    h1, h2, h3, p {
        color: #212529 !important;
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
        st.markdown("**⚡ 识别速度 | 📖 完整度**")
        
        for i in range(1, 6):
            times = st.session_state.step_times[i]
            covs = st.session_state.step_coverage[i]
            avg_time = sum(times) / len(times) if times else None
            avg_cov = (sum(covs) / len(covs) * 100) if covs else None
            
            time_str = f"{avg_time:.0f}s" if avg_time else "--s"
            cov_str = f"{avg_cov:.0f}%" if avg_cov else "--%"
            
            st.markdown(f"**{STEPS[i]['name']}**<br><span style='font-size:13px;'>{time_str} | {cov_str}</span>", unsafe_allow_html=True)
        
        st.markdown("---")
        st.markdown("**🧠 读盘画像**")
        if st.session_state.reading_profile:
            for step_key, weaknesses in st.session_state.reading_profile.items():
                for w, count in weaknesses.items():
                    if count >= 2:
                        step_name = STEPS[int(step_key)]['name'] if step_key.isdigit() else step_key
                        st.markdown(f"⚠️ {step_name}: {w[:12]}… ×{count}")
        else:
            st.caption("暂无数据，完成一次复盘后显示")
        
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
            new_bar = st.slider("📌 K线位置", 60, max_bar, st.session_state.current_bar)
            if new_bar != st.session_state.current_bar:
                st.session_state.current_bar = new_bar
                analyzer = StructureAnalyzer(df, new_bar, lookback=80)
                st.session_state.structure_report, st.session_state.detected_elements = analyzer.generate_full_report()
                reset_step_progress()
                st.session_state.practice_start_time = time.time()
                st.session_state.current_step_start_time = time.time()
                st.rerun()

            col1, col2 = st.columns(2)
            if col1.button("🎲 随机", use_container_width=True):
                new_bar = random.randint(80, max_bar - 20)
                st.session_state.current_bar = new_bar
                analyzer = StructureAnalyzer(df, new_bar, lookback=80)
                st.session_state.structure_report, st.session_state.detected_elements = analyzer.generate_full_report()
                reset_step_progress()
                st.session_state.practice_start_time = time.time()
                st.session_state.current_step_start_time = time.time()
                st.rerun()

            if col2.button("🔄 重置", use_container_width=True):
                reset_step_progress()
                st.session_state.current_step_start_time = time.time()
                st.rerun()

            st.caption(f"K{st.session_state.current_bar} / {len(df)}根")

    # 主界面
    if st.session_state.df is None:
        st.markdown("## 👈 请从左侧选择品种开始训练")
        st.markdown("""
        <div style="background:#f8f9fa;padding:20px;border-radius:12px;border:1px solid #e9ecef;">
        <h3>Al Brooks 5步训练器 V4.4</h3>
        <p>AI会先独立分析图表结构，再批改你的观察。</p>
        <ul>
            <li><strong>第1步</strong>：画线 - 通道、楔形、超出</li>
            <li><strong>第2步</strong>：形态 - 双顶/双底、三角形、推进波</li>
            <li><strong>第3步</strong>：特殊K线 - IB、OB、IOI、大K线</li>
            <li><strong>第4步</strong>：值得做吗 - 概率、空间、风险</li>
            <li><strong>第5步</strong>：管理仓位 - 退出、移动止损</li>
        </ul>
        <p style="color:#6c757d;">💡 画线没有唯一正确答案，灵活比完美更重要。</p>
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
    col3.progress(completed_count / 5, text=f"{completed_count}/5步")

    fig = build_chart(df, bar, step_name=current_step["name"])
    st.plotly_chart(fig, use_container_width=True)

    # 全部完成
    if all(st.session_state.step_completed.values()):
        st.success("🎉 完成一次完整复盘！")
        
        # 记录本次复盘
        if st.session_state.practice_start_time:
            elapsed = time.time() - st.session_state.practice_start_time
            st.metric("本次总耗时", f"{elapsed:.0f}秒")
            st.session_state.total_practices += 1  # 增加完成次数
        
        st.markdown("**本次详情：**")
        cols = st.columns(5)
        for i in range(1, 6):
            times = st.session_state.step_times[i]
            coverages = st.session_state.step_coverage[i]
            last_time = times[-1] if times else 0
            last_cov = coverages[-1] * 100 if coverages else 0
            cols[i-1].metric(STEPS[i]['name'], f"{last_time:.0f}s", f"{last_cov:.0f}%")
        
        if st.button("🔄 下一次复盘", type="primary"):
            new_bar = random.randint(80, len(df) - 20)
            st.session_state.current_bar = new_bar
            analyzer = StructureAnalyzer(df, new_bar, lookback=80)
            st.session_state.structure_report, st.session_state.detected_elements = analyzer.generate_full_report()
            reset_step_progress()
            st.session_state.practice_start_time = time.time()
            st.session_state.current_step_start_time = time.time()
            st.rerun()
        return

    # 当前步骤
    st.markdown("---")
    st.markdown(f"## {current_step['name']} — {current_step['title']}")
    st.markdown(f"*{current_step['question']}*")

    with st.expander("📖 本步骤参考", expanded=False):
        for ce in current_step["core_elements"]:
            st.markdown(f"- {ce}")
        st.success(f"✅ {current_step['example_good']}")
        st.warning(f"⚠️ {current_step['example_bad']}")

    with st.expander("🤖 AI独立分析结果", expanded=False):
        st.code(st.session_state.structure_report or "未生成", language=None)

    # 对话
    conv = st.session_state.conversations[current_step_num]
    for msg in conv:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if st.session_state.step_completed[current_step_num]:
        if st.session_state.current_step_start_time:
            elapsed = time.time() - st.session_state.current_step_start_time
            st.session_state.step_times[current_step_num].append(elapsed)
        st.success(f"✅ {current_step['name']} 完成！")
        if current_step_num < 5:
            time.sleep(1.5)
            st.session_state.step = current_step_num + 1
            st.session_state.current_step_start_time = time.time()
            st.rerun()
        return

    round_num = st.session_state.round_count[current_step_num]
    
    if round_num < MAX_ROUNDS:
        placeholder = "请描述你的观察..." if round_num == 0 else "可以补充..."
        user_input = st.chat_input(placeholder)

        if user_input:
            conv.append({"role": "user", "content": user_input})
            st.session_state.user_answers[current_step_num] += f"\n[第{round_num+1}轮] {user_input}"
            st.session_state.round_count[current_step_num] += 1

            with st.spinner("AI分析中..."):
                coach_reply = call_coach(
                    step_num=current_step_num,
                    conversation_history=conv,
                    structure_report=st.session_state.structure_report,
                    detected_elements=st.session_state.detected_elements,
                    reading_profile=st.session_state.reading_profile,
                    round_num=round_num + 1,
                    max_rounds=MAX_ROUNDS,
                )

            conv.append({"role": "assistant", "content": coach_reply})
            
            if "[NEXT]" in coach_reply or round_num + 1 >= MAX_ROUNDS:
                st.session_state.step_completed[current_step_num] = True
                summary = summarize_step(current_step_num, st.session_state.user_answers[current_step_num])
                st.session_state.step_summaries[current_step_num] = summary
                
                # 计算完整度
                covered, total, ratio = calculate_coverage(
                    st.session_state.user_answers[current_step_num],
                    st.session_state.detected_elements
                )
                st.session_state.step_coverage[current_step_num].append(ratio)
                
                # 更新薄弱点
                if ratio < 0.6:
                    step_key = str(current_step_num)
                    if step_key not in st.session_state.reading_profile:
                        st.session_state.reading_profile[step_key] = {}
                    st.session_state.reading_profile[step_key]["观察不完整"] = \
                        st.session_state.reading_profile[step_key].get("观察不完整", 0) + 1
                
            st.rerun()


if __name__ == "__main__":
    main()
