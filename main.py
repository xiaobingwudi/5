"""
Al Brooks 日内机会寻找训练器 V4.3
核心增强：AI 先独立分析图表结构，再批改用户观察

新增分析能力：
1. 通道识别（上升/下降/水平通道）
2. 楔形检测（3个以上同向接触点）
3. 超出(Overshoot)检测
4. 双顶/双底增强
5. 收缩/扩展三角形检测
6. 推进波识别（一推、二推、三推）
7. 完整度计算基于实际存在的形态（而非固定清单）
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
import re
from collections import deque
from sklearn.linear_model import LinearRegression

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
        "core_elements": [
            "通道方向（上升/下降/水平）",
            "通道边界（至少2个高点+2个低点）",
            "楔形（3个以上同向接触点）",
            "超出(Overshoot)：K线突破通道边界",
            "备选画法和最终选择"
        ],
        "example_good": "我用K5、K15、K25的高点画了一条下降趋势线。K8、K18的低点构成了平行支撑线，形成一个下降通道。K18的低点向下刺穿了支撑线形成Overshoot，但很快收回。",
        "example_bad": "这是一个下降通道。",
    },
    2: {
        "name": "第2步：形态",
        "title": "识别双顶/双底、三角形等反转形态",
        "question": "图表上出现了哪些反转或延续形态？",
        "core_elements": [
            "双顶或双底",
            "收缩三角形（低点抬高+高点降低）",
            "扩展三角形（低点降低+高点抬高）",
            "一推、二推、三推",
            "失败突破"
        ],
        "example_good": "K12和K28构成更高低点的双底。K15-K20之间是收缩三角形。K22向上假突破后回落。",
        "example_bad": "有双底，要涨了。",
    },
    3: {
        "name": "第3步：特殊K线",
        "title": "识别信号K线",
        "question": "有哪些特殊的单根K线或K线组合？",
        "core_elements": [
            "大K线（实体>80%）",
            "惊讶K线（波幅>近期平均1.5倍）",
            "内包线(IB)",
            "外包线(OB)",
            "IOI模式",
            "连续强势K线"
        ],
        "example_good": "K15是大阳线，实体92%，高位收盘。K16是内包线。K17是外包线，形成IOI。K22是惊讶K线，波幅是平均的2倍。",
        "example_bad": "K15是大K线。",
    },
    4: {
        "name": "第4步：值得做吗",
        "title": "评估概率、空间和风险",
        "question": "这笔交易值得做吗？",
        "core_elements": [
            "方向（多头/空头）",
            "成功概率（高/中/低）",
            "空间至少2R",
            "失败代价",
            "是否值得"
        ],
        "example_good": "空头机会。双顶+下降通道，概率高。风险20点，目标40点以上。值得做。",
        "example_bad": "感觉要涨。",
    },
    5: {
        "name": "第5步：管理仓位",
        "title": "持仓期间的管理计划",
        "question": "入场后如何管理？",
        "core_elements": [
            "仓位规模",
            "提前退出条件",
            "移动止损",
            "分批止盈"
        ],
        "example_good": "2%仓位。若3根K线反向则离场。涨到1R后移动止损保本。",
        "example_bad": "拿着等涨。",
    },
}

# ==================== 增强的结构分析类 ====================
class StructureAnalyzer:
    """独立分析图表结构，不依赖AI视觉"""
    
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
        """识别摆动高低点"""
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
        """检测通道方向"""
        if len(self.swing_highs) < 2 or len(self.swing_lows) < 2:
            return "数据不足"
        
        highs = self.swing_highs[-3:]
        lows = self.swing_lows[-3:]
        
        if len(highs) >= 2:
            high_slope = (highs[-1]["price"] - highs[-2]["price"]) / (highs[-1]["idx"] - highs[-2]["idx"]) if highs[-1]["idx"] != highs[-2]["idx"] else 0
        else:
            high_slope = 0
        
        if len(lows) >= 2:
            low_slope = (lows[-1]["price"] - lows[-2]["price"]) / (lows[-1]["idx"] - lows[-2]["idx"]) if lows[-1]["idx"] != lows[-2]["idx"] else 0
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
        """检测楔形（3个以上同向接触点）"""
        if len(self.swing_highs) >= 3:
            x_highs = [[h["idx"]] for h in self.swing_highs[-3:]]
            y_highs = [h["price"] for h in self.swing_highs[-3:]]
            if len(x_highs) >= 2:
                try:
                    reg = LinearRegression().fit(x_highs, y_highs)
                    r2_high = reg.score(x_highs, y_highs)
                except:
                    r2_high = 0
            else:
                r2_high = 0
        else:
            r2_high = 0
        
        if len(self.swing_lows) >= 3:
            x_lows = [[l["idx"]] for l in self.swing_lows[-3:]]
            y_lows = [l["price"] for l in self.swing_lows[-3:]]
            if len(x_lows) >= 2:
                try:
                    reg = LinearRegression().fit(x_lows, y_lows)
                    r2_low = reg.score(x_lows, y_lows)
                except:
                    r2_low = 0
            else:
                r2_low = 0
        else:
            r2_low = 0
        
        wedge_detected = (r2_high > 0.9 and len(self.swing_highs) >= 3) or (r2_low > 0.9 and len(self.swing_lows) >= 3)
        return wedge_detected
    
    def detect_double_top_bottom(self):
        """检测双顶/双底"""
        results = []
        if len(self.swing_highs) >= 2:
            h1, h2 = self.swing_highs[-2], self.swing_highs[-1]
            diff_pct = abs(h2["price"] - h1["price"]) / h1["price"] if h1["price"] > 0 else 1
            if diff_pct < 0.02:
                if h2["price"] > h1["price"]:
                    results.append(f"更高高点双顶")
                else:
                    results.append(f"双顶")
        
        if len(self.swing_lows) >= 2:
            l1, l2 = self.swing_lows[-2], self.swing_lows[-1]
            diff_pct = abs(l2["price"] - l1["price"]) / l1["price"] if l1["price"] > 0 else 1
            if diff_pct < 0.02:
                if l2["price"] > l1["price"]:
                    results.append(f"更高低点双底")
                else:
                    results.append(f"双底")
        
        return results
    
    def detect_triangle(self):
        """检测三角形（收缩/扩展）"""
        if len(self.swing_highs) < 2 or len(self.swing_lows) < 2:
            return None
        
        high_diff = self.swing_highs[-1]["price"] - self.swing_highs[-2]["price"]
        low_diff = self.swing_lows[-1]["price"] - self.swing_lows[-2]["price"]
        
        if high_diff < 0 and low_diff > 0:
            return "收缩三角形（低点抬高，高点降低）"
        elif high_diff > 0 and low_diff < 0:
            return "扩展三角形（低点降低，高点抬高）"
        return None
    
    def detect_pushes(self):
        """检测推进波（一推、二推、三推）"""
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
        """检测特殊K线"""
        patterns = []
        ranges = []
        
        for i in range(max(0, len(self.sub)-20), len(self.sub)):
            row = self.sub.iloc[i]
            ranges.append(row["high"] - row["low"])
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
                patterns.append({"type": "大K线", "idx": orig_idx, "detail": f"{direction}({body_ratio*100:.0f}%)"})
            
            if total > avg_range * 1.5 and avg_range > 0:
                patterns.append({"type": "惊讶K线", "idx": orig_idx, "detail": f"波幅{total:.0f}，平均{avg_range:.0f}"})
            
            if curr["high"] < prev["high"] and curr["low"] > prev["low"]:
                patterns.append({"type": "IB", "idx": orig_idx})
            
            elif curr["high"] > prev["high"] and curr["low"] < prev["low"]:
                patterns.append({"type": "OB", "idx": orig_idx})
            
            if cc > co:
                if cc > curr["high"] - total * 0.2:
                    patterns.append({"type": "高位收盘", "idx": orig_idx})
            else:
                if cc < curr["low"] + total * 0.2:
                    patterns.append({"type": "低位收盘", "idx": orig_idx})
        
        for i in range(2, len(self.sub)):
            k1 = self.sub.iloc[i-2]
            k2 = self.sub.iloc[i-1]
            k3 = self.sub.iloc[i]
            if (k2["high"] < k1["high"] and k2["low"] > k1["low"]) and \
               (k3["high"] > k2["high"] and k3["low"] < k2["low"]):
                patterns.append({"type": "IOI", "idx": self.original_indices[i]})
        
        streak = 1
        for i in range(len(self.sub)-2, -1, -1):
            curr = self.sub.iloc[i]
            next_bar = self.sub.iloc[i+1]
            if (curr["close"] >= curr["open"]) == (next_bar["close"] >= next_bar["open"]):
                overlap = min(curr["high"], next_bar["high"]) - max(curr["low"], next_bar["low"])
                if overlap <= 0:
                    streak += 1
                else:
                    break
            else:
                break
        if streak >= 3:
            patterns.append({"type": "连续强势K线", "idx": self.original_indices[-1], "detail": f"{streak}根"})
        
        return patterns
    
    def generate_full_report(self):
        """生成完整的结构分析报告（AI独立分析结果）"""
        channel = self.detect_channel()
        wedge = self.detect_wedge()
        double_patterns = self.detect_double_top_bottom()
        triangle = self.detect_triangle()
        pushes = self.detect_pushes()
        special_bars = self.detect_special_bars()
        
        detected_elements = []
        
        report_lines = [
            "═══════ AI独立分析结果 ═══════",
            f"分析范围: K{self.start} ~ K{self.bar}（共{len(self.sub)}根K线）",
            "",
            "【第1步：通道和楔形】",
            f"  通道: {channel}",
        ]
        if wedge:
            report_lines.append("  楔形: ✓ 检测到")
            detected_elements.append("楔形")
        else:
            report_lines.append("  楔形: ✗ 未检测到")
        
        report_lines.append("")
        report_lines.append("【第2步：形态】")
        if double_patterns:
            report_lines.append(f"  双顶/双底: {', '.join(double_patterns)}")
            detected_elements.extend(double_patterns)
        else:
            report_lines.append("  双顶/双底: 未检测到")
        
        if triangle:
            report_lines.append(f"  三角形: {triangle}")
            detected_elements.append("三角形")
        else:
            report_lines.append("  三角形: 未检测到")
        
        if pushes:
            report_lines.append(f"  推进波: {', '.join(pushes)}")
            detected_elements.extend(pushes)
        else:
            report_lines.append("  推进波: 未检测到")
        
        report_lines.append("")
        report_lines.append("【第3步：特殊K线】")
        
        bar_types = {}
        for bar in special_bars:
            t = bar["type"]
            if t not in bar_types:
                bar_types[t] = []
            bar_types[t].append(bar["idx"])
        
        for t, indices in bar_types.items():
            idx_str = f"K{','.join(map(str, indices[:3]))}" + ("..." if len(indices) > 3 else "")
            report_lines.append(f"  {t}: {idx_str}")
            if t not in ["高位收盘", "低位收盘"]:
                detected_elements.append(t)
        
        report_lines.append("")
        report_lines.append("【第4步：值得做吗】")
        if double_patterns or wedge or triangle or pushes:
            report_lines.append("  潜在机会: 存在多重信号")
        else:
            report_lines.append("  潜在机会: 信号不明显")
        
        report_lines.append("")
        report_lines.append("【第5步：管理参考】")
        report_lines.append("  止损: 通常设在信号K线高低点外侧")
        report_lines.append("  目标: 至少2R")
        
        return "\n".join(report_lines), detected_elements


# ==================== AI 提示词 ====================
COACH_SYSTEM = """你是 Al Brooks 价格行为分析教练。

【核心原则】
- 你已经独立完成了图表的结构分析（见下方的【AI独立分析结果】）
- 你的任务是：基于你自己的分析结果，批改用户的观察
- 你不是在猜市场，你是有依据的

【批改规则】
1. 用户遗漏了实际存在的形态 → 追问
2. 用户说了实际不存在的形态 → 友善指出，不过度纠正
3. 用户判断与你的分析不一致但逻辑合理 → 认可灵活性

【反证原则】
- 如果用户给出判断，追问："什么情况会证明你是错的？"

【画线灵活性】
- 通道画法可以有多种合理方式
- 如果用户画法与你的分析不同但逻辑成立，不要纠正

【输出格式】
- 反馈控制在120字以内
- 如果用户已覆盖主要形态，输出"[NEXT]"
- 否则输出"[CONTINUE]"

【AI独立分析结果】（这是你自己的分析，已包含第1-5步的结论）
{structure_report}

【实际存在的形态】（用于检查用户是否遗漏）
{detected_elements}

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


def build_profile_guidance(reading_profile):
    if not reading_profile:
        return "暂无"
    items = []
    for step_key, weaknesses in reading_profile.items():
        for w, count in weaknesses.items():
            if count >= 2:
                items.append(f"第{step_key}步用户常漏：{w}")
    return "\n".join(items) if items else "无明显薄弱点"


def build_previous_findings(step_summaries):
    if not step_summaries:
        return "这是第1步"
    return "\n".join(f"第{k}步：{v}" for k, v in step_summaries.items())


# ==================== AI 调用函数（适配您的配置）====================
def _gpt(messages):
    """调用 DeepSeek API（适配 Streamlit Cloud 配置）"""
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


def calculate_coverage(user_answer, detected_elements):
    """基于实际存在的形态计算完整度"""
    if not detected_elements:
        return 0, 0, 1.0
    
    user_lower = user_answer.lower()
    covered = 0
    
    keyword_map = {
        "楔形": ["楔形", "wedge"],
        "双顶": ["双顶", "double top"],
        "双底": ["双底", "double bottom"],
        "三角形": ["三角形", "triangle"],
        "推进波": ["推", "push"],
        "大K线": ["大阳", "大阴", "大k"],
        "惊讶K线": ["惊讶", "surprise"],
        "IB": ["内包", "ib"],
        "OB": ["外包", "ob"],
        "IOI": ["ioi"],
        "连续强势K线": ["连续", "强势", "无重叠"],
    }
    
    for element in detected_elements:
        keywords = keyword_map.get(element, [element.lower()])
        if any(kw in user_lower for kw in keywords):
            covered += 1
    
    ratio = covered / len(detected_elements) if detected_elements else 1.0
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
        "structure_report": None,
        "detected_elements": [],
        "conversations": {i: [] for i in range(1, 6)},
        "user_answers": {i: "" for i in range(1, 6)},
        "step_completed": {i: False for i in range(1, 6)},
        "step_summaries": {},
        "reading_profile": {},
        "round_count": {i: 0 for i in range(1, 6)},
        "total_practices": 0,
        "step_times": {i: [] for i in range(1, 6)},
        "step_coverage": {i: [] for i in range(1, 6)},
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


def get_step_avg_time(step_num):
    times = st.session_state.step_times[step_num]
    if not times:
        return None
    return sum(times) / len(times)


def get_step_avg_coverage(step_num):
    coverages = st.session_state.step_coverage[step_num]
    if not coverages:
        return None
    return sum(coverages) / len(coverages) * 100


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
    st.set_page_config(page_title="Al Brooks 5步训练器", layout="wide")
    
    st.markdown("""
    <style>
    .stApp { background-color: #f5f7fa; }
    [data-testid="stSidebar"] { background-color: #1e293b; }
    [data-testid="stSidebar"] .stMarkdown, [data-testid="stSidebar"] .stMarkdown p {
        color: #e2e8f0 !important;
    }
    [data-testid="stSidebar"] .stMetric { background-color: #334155; border-radius: 8px; padding: 8px; }
    [data-testid="stSidebar"] .stButton button { background-color: #4f8ef7; color: white; border-radius: 6px; }
    [data-testid="stSidebar"] details { background-color: #334155; border-radius: 8px; padding: 4px 8px; }
    .stExpander { background-color: #ffffff; border-radius: 8px; border: 1px solid #e2e8f0; }
    .stChatMessage { background-color: #ffffff; border-radius: 12px; padding: 8px 12px; }
    .stSuccess { background-color: #dcfce7; color: #166534; }
    .coverage-high { color: #22c55e; }
    .coverage-mid { color: #f59e0b; }
    .coverage-low { color: #ef4444; }
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
            avg_time = get_step_avg_time(i)
            avg_cov = get_step_avg_coverage(i)
            trend = get_step_trend(i)
            
            time_str = f"{avg_time:.0f}s" if avg_time else "--s"
            trend_symbol = trend if trend else ""
            
            if avg_cov:
                cov_class = "coverage-high" if avg_cov >= 70 else ("coverage-mid" if avg_cov >= 50 else "coverage-low")
                cov_str = f'<span class="{cov_class}">{avg_cov:.0f}%</span>'
            else:
                cov_str = "--%"
            
            st.markdown(f"**{STEPS[i]['name']}**<br><span style='font-size:13px;'>{time_str}{trend_symbol} | {cov_str}</span>", unsafe_allow_html=True)
        
        st.markdown("---")
        st.markdown("**🧠 读盘画像**")
        if st.session_state.reading_profile:
            for step_key, weaknesses in st.session_state.reading_profile.items():
                for w, count in weaknesses.items():
                    if count >= 2:
                        st.markdown(f"⚠️ {STEPS[int(step_key)]['name']}: {w[:12]}… ×{count}")
        else:
            st.caption("暂无")
        
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
        <div style="background:#fff;padding:20px;border-radius:12px;border:1px solid #e2e8f0;">
        <h3>Al Brooks 5步训练器 V4.3</h3>
        <p>AI会先独立分析图表结构，再批改你的观察。</p>
        <ul>
            <li><strong>第1步</strong>：画线 - 通道、楔形、超出</li>
            <li><strong>第2步</strong>：形态 - 双顶/双底、三角形、推进波</li>
            <li><strong>第3步</strong>：特殊K线 - IB、OB、IOI、大K线</li>
            <li><strong>第4步</strong>：值得做吗 - 概率、空间、风险</li>
            <li><strong>第5步</strong>：管理仓位 - 退出、移动止损</li>
        </ul>
        <p style="color:#64748b;">💡 画线没有唯一正确答案，灵活比完美更重要。</p>
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
        if st.session_state.practice_start_time:
            elapsed = time.time() - st.session_state.practice_start_time
            st.metric("本次总耗时", f"{elapsed:.0f}秒")
        
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
            st.session_state.total_practices += 1
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

    with st.expander("🤖 AI独立分析结果（仅用于批改，不直接抄答案）", expanded=False):
        st.code(st.session_state.structure_report or "未生成", language=None)
        st.caption(f"当前图表实际存在的形态：{', '.join(st.session_state.detected_elements) if st.session_state.detected_elements else '无明显形态'}")

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
    max_rounds = MAX_ROUNDS
    
    if round_num < max_rounds:
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
                    step_summaries=st.session_state.step_summaries,
                    reading_profile=st.session_state.reading_profile,
                    round_num=round_num + 1,
                    max_rounds=max_rounds,
                )

            conv.append({"role": "assistant", "content": coach_reply})
            
            if "[NEXT]" in coach_reply or round_num + 1 >= max_rounds:
                st.session_state.step_completed[current_step_num] = True
                summary = summarize_step(current_step_num, st.session_state.user_answers[current_step_num])
                st.session_state.step_summaries[current_step_num] = summary
                
                covered, total, ratio = calculate_coverage(
                    st.session_state.user_answers[current_step_num],
                    st.session_state.detected_elements
                )
                st.session_state.step_coverage[current_step_num].append(ratio)
                
                if ratio < 0.6:
                    step_key = str(current_step_num)
                    if step_key not in st.session_state.reading_profile:
                        st.session_state.reading_profile[step_key] = {}
                    st.session_state.reading_profile[step_key]["观察不完整"] = \
                        st.session_state.reading_profile[step_key].get("观察不完整", 0) + 1
                
            st.rerun()


if __name__ == "__main__":
    main()
