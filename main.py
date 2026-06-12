"""
Al Brooks 五步决策训练器 V4.1
核心重构：从"形态识别"转向"实时决策能力"

关键改进（基于用户反馈）：
1. 市场验证改为"假设存活时间"而非"涨跌对错"
2. Step3 从固定1.5R改为期望值（胜率×回报）思维
3. Signal批改增加"失败意味着什么"维度
4. 验证长度随机化（6-15根），用户不知道答案何时揭晓

Brooks真正训练的是决策过程，不是形态命名：
Step1: Always In? — 现在必须持仓，选多还是空？
Step2: Market State — 当前环境适合做什么？
Step3: Expected Value — 值不值得赌？
Step4: Signal — 如果失败说明什么？
Step5: Trade Plan — 做/不做/条件做，放弃条件
验证: 假设存活了多久？什么时候必须换边？
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import akshare as ak
from openai import OpenAI
import random, time
from datetime import datetime

# ==================== 配置 ====================
SYMBOL_NAMES = {
    "IF": "沪深300股指", "IH": "上证50股指", "IC": "中证500股指", "IM": "中证1000股指",
    "CU": "沪铜", "AL": "沪铝", "ZN": "沪锌", "AU": "黄金", "AG": "白银",
    "RB": "螺纹钢", "I": "铁矿石", "J": "焦炭", "JM": "焦煤",
    "MA": "甲醇", "TA": "PTA", "SA": "纯碱", "FG": "玻璃",
    "A": "豆一", "M": "豆粕", "Y": "豆油", "C": "玉米",
    "CF": "棉花", "SR": "白糖", "SC": "原油",
}
EXCHANGES = {
    "股指": ["IF", "IH", "IC", "IM"],
    "黑色": ["RB", "I", "J", "JM"],
    "化工": ["MA", "TA", "SA", "FG"],
    "农产品": ["A", "M", "Y", "C", "CF", "SR"],
    "能源": ["SC"],
}
PERIODS = {"5": "5分钟", "15": "15分钟", "30": "30分钟", "60": "60分钟"}

# 五步标签（用于报告追踪）
FIVE_STEPS = ["Always In", "Market State", "Expected Value", "Signal", "Trade Plan"]

# Always In信心等级
CONFIDENCE_LEVELS = ["51%（勉强偏向）", "55%", "60%", "70%", "80%以上（强烈确信）"]

# 验证选项（三选一）
VER_OPTIONS = ["✅ 交易假设维持（逻辑仍然成立）",
               "⚠️ 交易假设减弱（仍持有但需观察）",
               "❌ 交易假设失效（需要退出或反向）"]

# 预期胜率选项（用于期望值计算）
WIN_RATE_OPTIONS = ["40%", "45%", "50%", "55%", "60%", "65%", "70%"]

# ==================== 结构预处理 ====================

def compute_bar_features(df):
    df = df.copy()
    df['body'] = abs(df['close'] - df['open'])
    df['total_range'] = df['high'] - df['low']
    df['body_ratio'] = np.where(df['total_range'] > 0, df['body'] / df['total_range'], 0)
    df['direction'] = np.where(df['close'] >= df['open'], 'bull', 'bear')
    df['upper_shadow'] = df['high'] - df[['open', 'close']].max(axis=1)
    df['lower_shadow'] = df[['open', 'close']].min(axis=1) - df['low']
    df['upper_ratio'] = np.where(df['total_range'] > 0, df['upper_shadow'] / df['total_range'], 0)
    df['lower_ratio'] = np.where(df['total_range'] > 0, df['lower_shadow'] / df['total_range'], 0)
    inside, outside = [], []
    for i in range(len(df)):
        if i == 0:
            inside.append(False); outside.append(False)
        else:
            ph, pl = df['high'].iloc[i-1], df['low'].iloc[i-1]
            ch, cl = df['high'].iloc[i], df['low'].iloc[i]
            inside.append(ch <= ph and cl >= pl)
            outside.append(ch > ph and cl < pl)
    df['is_inside'] = inside
    df['is_outside'] = outside
    avg_body = df['body'].rolling(20, min_periods=3).mean()
    df['is_big'] = df['body'] > avg_body * 1.5
    return df


def find_swings(df, order=3):
    high = df['high'].values
    low  = df['low'].values
    n = len(high)
    sh, sl = [], []
    for i in range(order, n - order):
        h_win = high[i-order: i+order+1]
        l_win = low[i-order:  i+order+1]
        if high[i] >= max(h_win):
            sh.append((i, float(high[i])))
        if low[i] <= min(l_win):
            sl.append((i, float(low[i])))
    def dedup(pts, cmp_fn):
        if not pts: return pts
        result = [pts[0]]
        for p in pts[1:]:
            if p[0] - result[-1][0] <= order:
                if cmp_fn(p[1], result[-1][1]):
                    result[-1] = p
            else:
                result.append(p)
        return result
    sh = dedup(sh, lambda a, b: a > b)
    sl = dedup(sl, lambda a, b: a < b)
    return sh, sl


def build_behavior_context(df, n_bars=60):
    if len(df) < 15: return None, {}
    data = df.tail(n_bars).copy().reset_index(drop=True)
    data = compute_bar_features(data)
    order = max(2, len(data) // 15)
    sh, sl = find_swings(data, order)

    # 高低点行为描述
    if len(sh) >= 2:
        sh_vals = [v for _, v in sh[-4:]]
        sh_dirs = []
        for i in range(1, len(sh_vals)):
            diff = sh_vals[i] - sh_vals[i-1]
            pct = diff / sh_vals[i-1] * 100
            sh_dirs.append(f"{'↑' if diff > 0 else '↓'}{abs(pct):.1f}%")
        sh_desc = "→".join(f"{v:.0f}" for v in sh_vals) + f"（{', '.join(sh_dirs)}）"
    else:
        sh_desc = "高点不足"

    if len(sl) >= 2:
        sl_vals = [v for _, v in sl[-4:]]
        sl_dirs = []
        for i in range(1, len(sl_vals)):
            diff = sl_vals[i] - sl_vals[i-1]
            pct = diff / sl_vals[i-1] * 100
            sl_dirs.append(f"{'↑' if diff > 0 else '↓'}{abs(pct):.1f}%")
        sl_desc = "→".join(f"{v:.0f}" for v in sl_vals) + f"（{', '.join(sl_dirs)}）"
    else:
        sl_desc = "低点不足"

    # 回调深度
    pullback_desc = "无法计算"
    if len(sh) >= 2 and len(sl) >= 2:
        last_up = sh[-1][1] - sl[-2][1] if sh[-1][0] > sl[-2][0] else None
        if last_up and sh[-2][1]:
            pb_ratio = (sh[-1][1] - sl[-1][1]) / (sh[-1][1] - sl[-2][1]) if (sh[-1][1] - sl[-2][1]) > 0 else 0
            pullback_desc = f"最近回调深度约{pb_ratio:.0%}（{'浅回调<38%' if pb_ratio < 0.38 else '深回调>50%' if pb_ratio > 0.5 else '中等回调38-50%'}）"

    # EMA关系
    ema20 = data['close'].ewm(span=20).mean().iloc[-1]
    price = data['close'].iloc[-1]
    ema_diff = price - ema20
    ema_pct = ema_diff / ema20 * 100
    if abs(ema_pct) < 0.3:
        ema_desc = f"价格紧贴EMA20（偏离{ema_pct:.1f}%）"
    elif ema_pct > 0:
        ema_desc = f"价格在EMA20上方{ema_pct:.1f}%"
    else:
        ema_desc = f"价格在EMA20下方{abs(ema_pct):.1f}%"

    # 近期K线详情
    recent = data.tail(8)
    bar_lines = []
    for i, row in recent.iterrows():
        d = "阳" if row['direction'] == 'bull' else "阴"
        sz = "大" if row['is_big'] else ("内包" if row['is_inside'] else ("外包" if row['is_outside'] else "普通"))
        bar_lines.append(f"K{i+1}: {sz}{d}线 | 实体{row['body_ratio']:.0%} | 上影{row['upper_ratio']:.0%} | 下影{row['lower_ratio']:.0%}")

    # 特殊K线
    specials = []
    recent6 = data.tail(6).reset_index(drop=True)
    streak = 0
    for v in reversed(recent6['is_inside'].tolist()):
        if v: streak += 1
        else: break
    if streak >= 2: specials.append(f"连续{streak}根内包（双方僵持）")
    bars6 = list(recent6.itertuples())
    for i in range(len(bars6)-2):
        if bars6[i].is_inside and bars6[i+1].is_outside and bars6[i+2].is_inside:
            specials.append("IOI形态（内-外-内，能量积蓄）"); break
    if recent6.iloc[-1]['is_outside']:
        pos = "收盘偏高（买方最终控制）" if recent6.iloc[-1]['close'] > recent6.iloc[-1]['open'] else "收盘偏低（卖方最终控制）"
        specials.append(f"最新K线外包——{pos}")
    if recent6.iloc[-1]['is_big']:
        d = "买方" if recent6.iloc[-1]['direction'] == 'bull' else "卖方"
        specials.append(f"最新K线是大K线（{d}强势推进）")
    if len(recent6) >= 2:
        p, c = recent6.iloc[-2], recent6.iloc[-1]
        if p['is_big'] and c['direction'] != p['direction'] and c['body_ratio'] > 0.5:
            specials.append("大K线后反向强K线——可能竭尽或反转")

    # ATR计算
    atr_n = min(14, len(data)-1)
    if atr_n > 0:
        tr = pd.concat([
            data['high'] - data['low'],
            (data['high'] - data['close'].shift()).abs(),
            (data['low']  - data['close'].shift()).abs()
        ], axis=1).max(axis=1)
        atr = tr.rolling(atr_n).mean().iloc[-1]
        atr_desc = f"{atr:.1f}点（{atr_n}根K线平均波幅）"
    else:
        atr = None
        atr_desc = "无法计算"

    # 价格位置
    if sh and sl:
        nh, nl = sh[-1][1], sl[-1][1]
        rng = nh - nl
        if rng > 0:
            r = (price - nl) / rng
            pos_desc = f"当前价格在近期区间{r:.0%}位置（高点{nh:.0f}，低点{nl:.0f}，幅度{rng:.0f}）"
        else:
            pos_desc = "价格在高低点重合区域"
    else:
        pos_desc = "无法确定相对位置"

    # ATR距离
    if sh and atr and atr > 0:
        dist_to_high = (sh[-1][1] - price) / atr
        dist_to_low  = (price - sl[-1][1]) / atr if sl else 0
        space_desc = f"距高点{dist_to_high:.1f}ATR，距低点{dist_to_low:.1f}ATR"
    else:
        space_desc = "无法计算ATR距离"

    context = f"""【高点序列】{sh_desc}
【低点序列】{sl_desc}
【回调】{pullback_desc}
【EMA】{ema_desc}
【位置】{pos_desc}
【ATR】{atr_desc}，{space_desc}
【最近8根】{chr(10).join(bar_lines)}
【特殊K线】{chr(10).join(specials) if specials else "无"}"""

    struct_data = {
        "swing_highs": sh, "swing_lows": sl,
        "price": price, "ema20": ema20, "atr": atr,
        "dist_to_high": (sh[-1][1] - price) if sh else None,
        "dist_to_low":  (price - sl[-1][1]) if sl else None,
    }
    return context, struct_data


# ==================== AI提示词 ====================

STEP1_GRADER = """你是Al Brooks，批改学员的Always In判断。

【图表行为】
{context}

【学员判断】
方向：{direction}
信心：{confidence}
理由：{reason}

【批改】
1. 方向：与高低点序列一致？
2. 信心：{confidence}合理吗？偏高还是偏低？
3. 反方：做反方向最有力的K线是哪根？

【裁决】correct / partial / wrong

控制在180字+裁决行。"""


STEP2_GRADER = """你是Al Brooks，批改学员的市场状态判断。

【图表行为】
{context}

【学员判断】
状态：{market_state}
证据：{evidence}
演化：{evolution}

【批改】
1. 状态判断准确吗？引用具体K线
2. 最容易误判的K线是哪根？
3. 这个状态对Signal选择的影响？

【裁决】correct / partial / wrong

控制在150字+裁决行。"""


STEP3_GRADER = """你是Al Brooks，批改学员的期望值判断。

【图表行为（含ATR）】
{context}

【学员判断】
这笔交易值不值得做？
理由：{reason}

【批改】
1. 位置空间够不够？（引用ATR量化）
2. 胜率×回报的期望值是多少？
3. 如果空间不足，正确做法是什么？

【裁决】correct / partial / wrong

控制在150字+裁决行。"""


STEP4_GRADER = """你是Al Brooks，批改学员的信号判断。

【图表行为】
{context}

【学员信号】
信号：{signal}
引用K线：{k_ref}

【批改标准】
① 与Always In方向一致？
② 出现在合理位置？
③ 提供足够期望值？
④ 如果失败说明什么？（维持/减弱/反转）

无信号选择放弃，是完全正确的答案。

【裁决】correct / partial / wrong

控制在150字+裁决行。"""


STEP5_GRADER = """你是Al Brooks，批改学员的交易计划。

【五步汇总】
Always In：{direction}（{confidence}）
市场状态：{market_state}
期望值：{ev_judgment}
信号：{signal}

【学员计划】
决策：{decision}
胜率预估：{win_rate}
放弃条件：{abandon}

【批改】
1. 四个条件都满足吗？
2. 胜率{win_rate}合理吗？
3. 放弃条件是否具体？
4. 最终裁决：做/不做/条件性做

【裁决】correct / partial / wrong

控制在200字+裁决行。"""


VERIFICATION_GRADER = """你是Al Brooks，批改学员的验证判断。

【原始假设】
Always In：{direction}（{confidence}）
放弃条件：{abandon}

【新K线】
{new_bars}

【学员判断】
状态：{ver_choice}
理由：{ver_reason}

【批改】
1. 假设存活了多久？失效还是延续？
2. 是否出现放弃条件？
3. 如果失效，原因是什么？（结构破坏/信号失败/反转信号）

【裁决】correct / partial / wrong

控制在150字+裁决行。"""


COACH_REPORT_PROMPT = """你是Al Brooks首席教练，写训练报告。

【数据】{symbol} {period}，{rounds}轮，{total_min:.0f}分钟
【五步得分】Always In:{s1}% Market State:{s2}% EV:{s3}% Signal:{s4}% Plan:{s5}%
【假设存活率】{survival:.0f}%
【错误】{errors}

【要求】
1. 最薄弱的1-2步及具体错误
2. 针对性建议（练什么，在什么结构下）
3. 最好的一步
4. 真诚鼓励

300字。"""


# ==================== AI调用 ====================

def get_client():
    key = st.secrets.get("OPENAI_API_KEY", "")
    url = st.secrets.get("OPENAI_BASE_URL", "https://api.deepseek.com")
    model = st.secrets.get("OPENAI_MODEL", "deepseek-chat")
    return OpenAI(base_url=url, api_key=key), model, bool(key)


def call_ai(prompt, max_tokens=400, temp=0.3):
    client, model, has_key = get_client()
    if not has_key:
        return "（未配置OPENAI_API_KEY）"
    try:
        r = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temp, max_tokens=max_tokens
        )
        return r.choices[0].message.content
    except Exception as e:
        return f"AI调用失败：{str(e)[:100]}"


def parse_grade_and_score(ai_response, step_name):
    for line in reversed(ai_response.strip().split("\n")):
        line = line.strip()
        if "【裁决】" in line:
            if "correct" in line:
                st.session_state.scores[step_name].append(1.0)
            elif "partial" in line:
                st.session_state.scores[step_name].append(0.5)
            elif "wrong" in line:
                st.session_state.scores[step_name].append(0.0)
            return
    st.session_state.scores[step_name].append(0.0)


# ==================== 数据加载 ====================

@st.cache_data(ttl=1800, show_spinner=False)
def load_data(symbol, period):
    try:
        df = ak.futures_zh_minute_sina(symbol=f"{symbol}0", period=period)
        if df is None or len(df) < 80: return None
        return df.rename(columns={
            "date": "time", "open": "open", "high": "high",
            "low": "low", "close": "close", "volume": "volume"
        }).reset_index(drop=True)
    except: return None


# ==================== 图表 ====================

def build_chart(df, obs_end, extra_bars=0, sh=None, sl=None):
    view_end = min(obs_end + extra_bars, len(df))
    display_n = 80
    start = max(0, view_end - display_n)
    data = df.iloc[start:view_end].copy().reset_index(drop=True)
    n = len(data)
    offset = start

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        vertical_spacing=0.02, row_heights=[0.76, 0.24])

    fig.add_trace(go.Candlestick(
        x=data.index, open=data['open'], high=data['high'],
        low=data['low'], close=data['close'], showlegend=False,
        increasing_line_color="#ef5350", decreasing_line_color="#26a69a",
    ), row=1, col=1)

    ema = data['close'].ewm(span=20).mean()
    fig.add_trace(go.Scatter(
        x=data.index, y=ema,
        line=dict(color="#ff9800", width=1.2, dash="dot"),
        name="EMA20", showlegend=True
    ), row=1, col=1)

    if sh:
        for idx, val in sh:
            pi = idx - offset
            if 0 <= pi < n:
                fig.add_annotation(x=pi, y=val, text="H",
                    font=dict(size=9, color="#ef5350"), showarrow=False, yshift=11, row=1, col=1)
    if sl:
        for idx, val in sl:
            pi = idx - offset
            if 0 <= pi < n:
                fig.add_annotation(x=pi, y=val, text="L",
                    font=dict(size=9, color="#26a69a"), showarrow=False, yshift=-13, row=1, col=1)

    obs_plot = obs_end - start - 1
    if 0 <= obs_plot < n:
        fig.add_vline(x=obs_plot, line_dash="dash",
                      line_color="#ff9800", line_width=1.5, opacity=0.8)
        fig.add_annotation(x=obs_plot, y=data['high'].max(),
            text="截止", font=dict(size=9, color="#ff9800"), showarrow=False, yshift=14)

    if extra_bars > 0 and obs_plot < n:
        fig.add_vrect(x0=max(0, obs_plot), x1=n-1,
                      fillcolor="#1976d2", opacity=0.06, layer="below", line_width=0)

    for i in range(4, n, 5):
        row_data = data.iloc[i]
        abs_num = i + offset + 1
        y = row_data['low'] if row_data['close'] >= row_data['open'] else row_data['high']
        shift = -13 if row_data['close'] >= row_data['open'] else 13
        fig.add_annotation(x=i, y=y, text=f"K{abs_num}",
            showarrow=False, font=dict(size=8, color="#999"), yshift=shift)

    vc = ["#ef5350" if c >= o else "#26a69a" for o, c in zip(data['open'], data['close'])]
    fig.add_trace(go.Bar(x=data.index, y=data['volume'],
                         marker_color=vc, showlegend=False, opacity=0.5), row=2, col=1)

    fig.update_layout(
        xaxis_rangeslider_visible=False, height=500,
        margin=dict(l=8, r=8, t=25, b=8),
        paper_bgcolor="#ffffff", plot_bgcolor="#fafafa",
        font=dict(color="#333"),
        legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="right", x=1, font=dict(size=11))
    )
    fig.update_xaxes(showgrid=True, gridcolor="#eeeeee", gridwidth=0.5,
                     zeroline=False, showticklabels=False, row=1, col=1)
    fig.update_xaxes(showgrid=True, gridcolor="#eeeeee", gridwidth=0.5, zeroline=False, row=2, col=1)
    fig.update_yaxes(showgrid=True, gridcolor="#eeeeee", gridwidth=0.5, zeroline=False)
    return fig


def get_new_bars_text(df, obs_end, n):
    lines = []
    for i in range(n):
        idx = obs_end + i
        if idx >= len(df): break
        row = df.iloc[idx]
        o, h, l, c = row['open'], row['high'], row['low'], row['close']
        body = abs(c - o); total = h - l
        br = body / total if total > 0 else 0
        d = "阳" if c >= o else "阴"
        special = ""
        if idx > 0:
            prev = df.iloc[idx-1]
            if h <= prev['high'] and l >= prev['low']: special = "（内包）"
            elif h > prev['high'] and l < prev['low']: special = "（外包）"
        lines.append(f"K{idx+1}{special}：{d}线 O={o:.0f} H={h:.0f} L={l:.0f} C={c:.0f} 实体{br:.0%}")
    return "\n".join(lines)


def compute_verification_score(df, obs_end, ver_total, direction, confidence):
    """
    市场验证：基于假设存活时间，不是涨跌对错
    """
    try:
        start_idx = obs_end
        end_idx = min(obs_end + ver_total, len(df)) - 1
        
        # 获取摆动点结构
        sub = df.iloc[:end_idx+1].copy()
        order = max(2, len(sub) // 15)
        sh, sl = find_swings(sub, order)
        sh = [s for s in sh if s[0] >= start_idx]
        sl = [s for s in sl if s[0] >= start_idx]
        
        is_long = "多" in direction
        
        # 检查放弃条件触发
        abandon_triggered = False
        abandon_reason = ""
        
        # 条件1：出现反向的LH/LL（结构破坏）
        if is_long:
            if len(sl) >= 2 and sl[-1][1] < sl[-2][1]:
                abandon_triggered = True
                abandon_reason = "出现更低低点（HL结构破坏）"
            elif len(sh) >= 2 and sh[-1][1] < sh[-2][1] and len(sh) >= 2:
                abandon_triggered = True
                abandon_reason = "出现更低高点（LH，空头开始控制）"
        else:
            if len(sh) >= 2 and sh[-1][1] > sh[-2][1]:
                abandon_triggered = True
                abandon_reason = "出现更高高点（HH结构破坏）"
            elif len(sl) >= 2 and sl[-1][1] > sl[-2][1] and len(sl) >= 2:
                abandon_triggered = True
                abandon_reason = "出现更高低点（HL，多头开始控制）"
        
        # 条件2：信号K线被吞没
        # 简化：最后3根K线中有外包反向K线
        last_bars = df.iloc[max(start_idx, end_idx-3):end_idx+1]
        if len(last_bars) >= 2:
            prev = last_bars.iloc[-2]
            curr = last_bars.iloc[-1]
            if curr['close'] > prev['high'] and is_long and curr['close'] > curr['open']:
                pass  # 上涨，不触发
            elif curr['close'] < prev['low'] and not is_long and curr['close'] < curr['open']:
                pass  # 下跌，不触发
            elif curr['high'] > prev['high'] and curr['low'] < prev['low']:
                # 外包线，方向与假设相反
                if (is_long and curr['close'] < curr['open']) or (not is_long and curr['close'] > curr['open']):
                    abandon_triggered = True
                    abandon_reason = "出现反向外包线（信号被吞没）"
        
        # 存活比例（基于时间）
        total_bars = min(ver_total, end_idx - start_idx + 1)
        if abandon_triggered:
            # 找到失效位置
            for i in range(start_idx, end_idx + 1):
                if i == start_idx: continue
                bar = df.iloc[i]
                if is_long and bar['low'] < df.iloc[i-1]['low']:
                    survival_bars = i - start_idx
                    break
                elif not is_long and bar['high'] > df.iloc[i-1]['high']:
                    survival_bars = i - start_idx
                    break
            else:
                survival_bars = total_bars
            survival_ratio = survival_bars / total_bars if total_bars > 0 else 0
        else:
            survival_ratio = 1.0
        
        # 评分
        if survival_ratio >= 0.8:
            score = 1.0
            desc = f"✅ 假设存活{survival_ratio:.0%}时间，验证期未触发放弃条件"
        elif survival_ratio >= 0.4:
            score = 0.5
            desc = f"⚠️ 假设存活{survival_ratio:.0%}时间，{abandon_reason}" if abandon_triggered else f"部分正确，存活{survival_ratio:.0%}"
        else:
            score = 0.0
            desc = f"❌ 假设迅速失效（存活{survival_ratio:.0%}），{abandon_reason}"
        
        return score, desc
    except Exception as e:
        return None, f"评分失败: {str(e)[:50]}"


# ==================== Session State ====================

def init_state():
    d = {
        "df": None, "symbol": None, "period": "15",
        "phase": "select",
        "obs_end": 0,
        "ver_total": 0,
        "context": "",
        "struct_data": {},
        # 五步输入
        "s1_dir": "", "s1_confidence": "", "s1_reason": "",
        "s2_state": "", "s2_evidence": "", "s2_evolution": "",
        "s3_reason": "",
        "s4_signal": "", "s4_kref": "",
        "s5_decision": "", "s5_winrate": "", "s5_abandon": "",
        # AI批改缓存
        "grade_s1": None, "grade_s2": None, "grade_s3": None,
        "grade_s4": None, "grade_s5": None, "grade_ver": None,
        # 五步得分
        "scores": {s: [] for s in FIVE_STEPS},
        "verification_scores": [],
        "round_errors": [],
        "round_num": 0,
        "round_start": None,
        "round_times": [],
        "coach_report": None,
        "practice_count": 0,
        "session_start": None,
    }
    for k, v in d.items():
        if k not in st.session_state:
            st.session_state[k] = v


def reset_round():
    for k in ["s1_dir","s1_confidence","s1_reason",
              "s2_state","s2_evidence","s2_evolution",
              "s3_reason",
              "s4_signal","s4_kref",
              "s5_decision","s5_winrate","s5_abandon",
              "grade_s1","grade_s2","grade_s3",
              "grade_s4","grade_s5","grade_ver"]:
        st.session_state[k] = None if k.startswith("grade") else ""
    st.session_state.round_start = time.time()


def load_sym(code, period):
    with st.spinner(f"加载 {code} {PERIODS[period]}..."):
        df = load_data(code, period)
    if df is None or len(df) < 100: return False
    st.session_state.df = df
    st.session_state.symbol = code
    st.session_state.period = period
    return True


def start_round(df, n_bars=60):
    # 随机验证长度 6-15 根
    ver_total = random.randint(6, 15)
    max_obs = len(df) - ver_total - 10
    min_obs = n_bars
    if max_obs < min_obs: return False
    obs_end = random.randint(min_obs, max_obs)
    st.session_state.obs_end = obs_end
    st.session_state.ver_total = ver_total
    result = build_behavior_context(df.iloc[:obs_end], n_bars)
    if result[0] is None: return False
    st.session_state.context = result[0]
    st.session_state.struct_data = result[1]
    st.session_state.round_num += 1
    reset_round()
    return True


def score_step(step_name, val):
    st.session_state.scores[step_name].append(val)


def get_step_accuracy(step_name):
    scores = st.session_state.scores.get(step_name, [])
    return int(np.mean(scores) * 100) if scores else 0


# ==================== CSS ====================
CSS = """
<style>
.stApp { background:#fff; color:#222; }
[data-testid="stSidebar"] { background:#f7f7f7 !important; border-right:1px solid #e8e8e8; }

.step-header {
    display:flex; align-items:center; gap:14px;
    background:#f0f4ff; border-left:3px solid #1a56db;
    border-radius:0 8px 8px 0;
    padding:12px 18px; margin-bottom:16px;
}
.step-num {
    background:#1a56db; color:#fff;
    font-size:13px; font-weight:700;
    border-radius:50%; width:28px; height:28px;
    display:flex; align-items:center; justify-content:center;
    font-family:monospace; flex-shrink:0;
}
.step-title { font-size:16px; font-weight:700; color:#1a3a8f; }
.step-desc  { font-size:12px; color:#666; margin-top:2px; }

.step-progress {
    display:flex; gap:6px; margin-bottom:16px;
}
.step-dot {
    flex:1; height:4px; border-radius:2px; background:#e8e8e8;
}
.step-dot.done    { background:#1a56db; }
.step-dot.current { background:#60a5fa; }

.ai-box {
    background:#f0f4ff; border:1px solid #93c5fd;
    border-radius:8px; padding:16px; margin:12px 0;
}
.ai-box-warn {
    background:#fff7ed; border:1px solid #fdba74;
    border-left:3px solid #f97316;
    border-radius:8px; padding:16px; margin:12px 0;
}
.ai-box-ok {
    background:#f0fdf4; border:1px solid #86efac;
    border-radius:8px; padding:16px; margin:12px 0;
}
.ai-label { font-size:11px; color:#888; margin-bottom:8px; letter-spacing:.05em; }
.ai-text  { color:#222; line-height:1.85; font-size:14px; }

.ver-maintain { background:#f0fdf4; border:1px solid #86efac; border-radius:6px; padding:10px 14px; }
.ver-weaken   { background:#fffbeb; border:1px solid #fcd34d; border-radius:6px; padding:10px 14px; }
.ver-failed   { background:#fef2f2; border:1px solid #fca5a5; border-radius:6px; padding:10px 14px; }

.score-grid { display:flex; gap:8px; margin:12px 0; flex-wrap:wrap; }
.score-card {
    flex:1; min-width:80px;
    background:#f7f7f7; border:1px solid #e8e8e8;
    border-radius:8px; padding:10px 8px; text-align:center;
}
.score-val { font-size:20px; font-weight:700; font-family:monospace; }
.score-lbl { font-size:10px; color:#888; margin-top:2px; }

.context-box {
    background:#fafafa; border:1px solid #e8e8e8; border-radius:6px;
    padding:12px; font-family:monospace; font-size:11px;
    color:#555; line-height:1.7; white-space:pre-wrap;
}

.stButton>button {
    border-radius:6px; border:1px solid #ddd;
    background:#f5f5f5; color:#333; font-weight:500; transition:all .15s;
}
.stButton>button:hover { background:#1a56db; border-color:#1a56db; color:#fff; }
.stTextArea textarea { background:#fafafa !important; color:#222 !important; border-color:#ddd !important; }
.stRadio label { color:#333 !important; }
div[data-testid="stExpander"] { background:#fafafa; border:1px solid #e8e8e8; border-radius:8px; }

.brooks-note {
    border-left:2px solid #f97316; padding:8px 14px;
    color:#888; font-style:italic; font-size:12px; margin:10px 0;
}
.round-tag {
    display:inline-block; background:#eef2ff; border:1px solid #a5b4fc;
    border-radius:20px; padding:2px 12px; font-size:12px;
    color:#1a3a8f; font-family:monospace;
}
</style>
"""


# ==================== 主界面 ====================

def render_step_header(num, title, desc, total=5):
    dots = ""
    for i in range(1, total+1):
        cls = "done" if i < num else ("current" if i == num else "")
        dots += f'<div class="step-dot {cls}"></div>'
    st.markdown(f"""
    <div class="step-progress">{dots}</div>
    <div class="step-header">
        <div class="step-num">{num}</div>
        <div>
            <div class="step-title">{title}</div>
            <div class="step-desc">{desc}</div>
        </div>
    </div>""", unsafe_allow_html=True)


def render_ai_box(content, style="normal"):
    cls = {"normal": "ai-box", "warn": "ai-box-warn", "ok": "ai-box-ok"}.get(style, "ai-box")
    label = {"normal": "🎯 AI批改", "warn": "⚠️ AI批改", "ok": "✅ AI批改"}.get(style, "🎯 AI批改")
    st.markdown(f"""
    <div class="{cls}">
        <div class="ai-label">{label}</div>
        <div class="ai-text">{content}</div>
    </div>""", unsafe_allow_html=True)


def main():
    st.set_page_config(page_title="Al Brooks 五步决策训练器", layout="wide")
    st.markdown(CSS, unsafe_allow_html=True)
    init_state()

    df = st.session_state.df
    phase = st.session_state.phase
    symbol = st.session_state.symbol
    period = st.session_state.period
    struct_data = st.session_state.struct_data
    obs_end = st.session_state.obs_end

    # 侧边栏
    with st.sidebar:
        st.markdown("## Al Brooks 训练器")
        st.caption("五步决策 · 假设存活验证")
        st.markdown("---")

        if st.session_state.round_num > 0:
            st.markdown("**五步能力**")
            for step in FIVE_STEPS:
                acc = get_step_accuracy(step)
                color = "#22c55e" if acc >= 70 else ("#f97316" if acc >= 40 else "#ef4444")
                st.markdown(f"""
                <div style="margin:5px 0;">
                    <div style="display:flex;justify-content:space-between;font-size:12px;">
                        <span style="color:#555;">{step}</span>
                        <span style="color:{color};">{acc}%</span>
                    </div>
                    <div style="background:#eee;border-radius:2px;height:4px;"><div style="background:{color};width:{max(4,acc)}%;height:4px;"></div></div>
                </div>""", unsafe_allow_html=True)
            
            if st.session_state.verification_scores:
                surv = np.mean(st.session_state.verification_scores) * 100
                st.markdown(f"**假设存活率**<br><span style='font-size:20px;font-weight:700;'>{surv:.0f}%</span>", unsafe_allow_html=True)

        st.markdown("---")
        period_sel = st.selectbox("周期", list(PERIODS.keys()), format_func=lambda x: PERIODS[x], index=1)

        for cat, codes in EXCHANGES.items():
            with st.expander(cat, expanded=False):
                cols = st.columns(2)
                for i, code in enumerate(codes):
                    if cols[i%2].button(code, key=f"s_{code}", use_container_width=True):
                        if load_sym(code, period_sel):
                            if start_round(st.session_state.df):
                                st.session_state.phase = "step1"
                                st.session_state.practice_count += 1
                                st.session_state.session_start = time.time()
                        st.rerun()

        st.markdown("---")
        if st.button("🎲 随机品种", use_container_width=True):
            code = random.choice([c for codes in EXCHANGES.values() for c in codes])
            if load_sym(code, period_sel):
                if start_round(st.session_state.df):
                    st.session_state.phase = "step1"
                    st.session_state.practice_count += 1
                    st.session_state.session_start = time.time()
            st.rerun()

        if phase not in ("select", "report"):
            st.markdown("---")
            if st.button("📊 结束出报告", use_container_width=True):
                st.session_state.phase = "report"
                st.rerun()
            if st.button("🔄 重置", use_container_width=True):
                for k in list(st.session_state.keys()): del st.session_state[k]
                st.rerun()

        st.markdown("""
        <div class="brooks-note">
        "The question is never what the market is doing. It's what you should do about it."<br>— Al Brooks
        </div>""", unsafe_allow_html=True)

    # 欢迎页
    if phase == "select":
        st.markdown("## 👈 从左侧选择品种开始")
        st.markdown("""
        <div style="background:#f0f4ff;border:1px solid #93c5fd;border-radius:12px;padding:28px;margin-top:16px;">
        <h3 style="color:#1a3a8f;">Al Brooks 五步决策训练器 V4.1</h3>
        <p>训练的不是"识别形态"，而是"在不确定中做决策"</p>
        <hr>
        <h4>五步决策流程</h4>
        <ul>
        <li><b>Step1 Always In</b> — 如果必须持仓，选多还是空？</li>
        <li><b>Step2 Market State</b> — 当前环境适合做什么？</li>
        <li><b>Step3 Expected Value</b> — 值不值得赌？</li>
        <li><b>Step4 Signal</b> — 如果失败说明什么？</li>
        <li><b>Step5 Trade Plan</b> — 做/不做/条件做，放弃条件</li>
        <li><b>验证</b> — 假设存活了多久？什么情况会失效？</li>
        </ul>
        <hr>
        <p>✦ 验证长度随机（6-15根），你不知道答案何时揭晓<br>
        ✦ 评分基于假设存活时间，不是涨跌对错</p>
        </div>""", unsafe_allow_html=True)
        return

    if df is None:
        st.warning("请先选择品种"); return

    sh = struct_data.get("swing_highs", [])
    sl = struct_data.get("swing_lows", [])

    # 顶栏
    c1, c2, c3 = st.columns([3, 1, 1])
    with c1: st.markdown(f"**{symbol}** · {SYMBOL_NAMES.get(symbol,'')} · {PERIODS[period]}")
    with c2: st.markdown(f'<span class="round-tag">第{st.session_state.round_num}轮</span>', unsafe_allow_html=True)
    with c3:
        if st.session_state.ver_total:
            st.markdown(f'<span style="color:#888;font-size:12px;">验证长度: {st.session_state.ver_total}根</span>', unsafe_allow_html=True)

    # ========== Step1 ==========
    if phase == "step1":
        render_step_header(1, "Always In", "如果必须持仓，你选多还是空？")
        st.plotly_chart(build_chart(df, obs_end, sh=sh, sl=sl), use_container_width=True)

        if st.session_state.round_num == 1:
            st.info('Brooks: "If you had to be in, would you rather be long or short?" 哪怕51%也要选。')

        with st.form("s1_form"):
            dir_choice = st.radio("方向（必选）:", ["做多", "做空"], horizontal=True)
            confidence = st.select_slider("信心等级:", options=CONFIDENCE_LEVELS)
            reason = st.text_area("理由（引用具体K线行为）:", height=90)
            if st.form_submit_button("提交", type="primary"):
                if reason:
                    st.session_state.s1_dir = dir_choice
                    st.session_state.s1_confidence = confidence
                    st.session_state.s1_reason = reason
                    with st.spinner("AI批改中..."):
                        grade = call_ai(STEP1_GRADER.format(
                            context=st.session_state.context,
                            direction=dir_choice, confidence=confidence, reason=reason
                        ), max_tokens=420)
                    st.session_state.grade_s1 = grade
                    parse_grade_and_score(grade, "Always In")
                    st.session_state.phase = "step2"
                    st.rerun()
                else:
                    st.warning("请填写理由")

    # ========== Step2 ==========
    elif phase == "step2":
        render_step_header(2, "Market State", "当前环境适合做什么？")
        st.plotly_chart(build_chart(df, obs_end, sh=sh, sl=sl), use_container_width=True)

        if st.session_state.grade_s1:
            render_ai_box(st.session_state.grade_s1)

        with st.form("s2_form"):
            market_state = st.radio("市场状态:", ["趋势环境", "区间环境"], horizontal=True)
            evidence = st.text_area("支持证据:", placeholder="例如：回调浅、被快速吸收...", height=80)
            evolution = st.text_area("如果延续，对你的交易意味着什么？", height=70)
            if st.form_submit_button("提交", type="primary"):
                if evidence and evolution:
                    st.session_state.s2_state = market_state
                    st.session_state.s2_evidence = evidence
                    st.session_state.s2_evolution = evolution
                    with st.spinner("AI批改中..."):
                        grade = call_ai(STEP2_GRADER.format(
                            context=st.session_state.context,
                            market_state=market_state, evidence=evidence, evolution=evolution
                        ), max_tokens=370)
                    st.session_state.grade_s2 = grade
                    parse_grade_and_score(grade, "Market State")
                    st.session_state.phase = "step3"
                    st.rerun()
                else:
                    st.warning("请填写")

    # ========== Step3 ==========
    elif phase == "step3":
        render_step_header(3, "Expected Value", "值不值得赌？")
        st.plotly_chart(build_chart(df, obs_end, sh=sh, sl=sl), use_container_width=True)

        if st.session_state.grade_s2:
            render_ai_box(st.session_state.grade_s2)

        sd = st.session_state.struct_data
        atr = sd.get("atr")
        if atr and atr > 0:
            st.info(f"**ATR参考:** {atr:.1f}点 | 距高点{sd.get('dist_to_high',0):.1f}点 | 距低点{sd.get('dist_to_low',0):.1f}点")

        with st.form("s3_form"):
            reason = st.text_area("这笔交易值不值得做？为什么？", height=100)
            if st.form_submit_button("提交", type="primary"):
                if reason:
                    st.session_state.s3_reason = reason
                    with st.spinner("AI批改中..."):
                        grade = call_ai(STEP3_GRADER.format(
                            context=st.session_state.context, reason=reason
                        ), max_tokens=370)
                    st.session_state.grade_s3 = grade
                    parse_grade_and_score(grade, "Expected Value")
                    st.session_state.phase = "step4"
                    st.rerun()
                else:
                    st.warning("请填写")

    # ========== Step4 ==========
    elif phase == "step4":
        render_step_header(4, "Signal", "如果失败说明什么？")
        st.plotly_chart(build_chart(df, obs_end, sh=sh, sl=sl), use_container_width=True)

        if st.session_state.grade_s3:
            render_ai_box(st.session_state.grade_s3)

        with st.form("s4_form"):
            has_signal = st.radio("是否有信号:", ["有信号", "没有信号"], horizontal=True)
            signal_desc = st.text_area("信号描述/放弃理由:", height=80)
            k_ref = st.text_input("引用的K线编号:" if "有" in (has_signal or "") else "", placeholder="例如：K38")
            if st.form_submit_button("提交", type="primary"):
                if signal_desc:
                    st.session_state.s4_signal = f"{has_signal}: {signal_desc}"
                    st.session_state.s4_kref = k_ref or "无"
                    with st.spinner("AI批改中..."):
                        grade = call_ai(STEP4_GRADER.format(
                            context=st.session_state.context,
                            signal=st.session_state.s4_signal, k_ref=k_ref or "无"
                        ), max_tokens=370)
                    st.session_state.grade_s4 = grade
                    parse_grade_and_score(grade, "Signal")
                    st.session_state.phase = "step5"
                    st.rerun()
                else:
                    st.warning("请填写")

    # ========== Step5 ==========
    elif phase == "step5":
        render_step_header(5, "Trade Plan", "做/不做/条件做，放弃条件")
        st.plotly_chart(build_chart(df, obs_end, sh=sh, sl=sl), use_container_width=True)

        if st.session_state.grade_s4:
            render_ai_box(st.session_state.grade_s4)

        with st.form("s5_form"):
            decision = st.radio("最终决策:", ["做", "不做", "条件性做"], horizontal=True)
            win_rate = st.select_slider("预估胜率:", options=WIN_RATE_OPTIONS, value="55%")
            abandon = st.text_area("放弃条件（什么情况下认为判断错了）:", height=80)
            if st.form_submit_button("提交", type="primary"):
                if abandon:
                    st.session_state.s5_decision = decision
                    st.session_state.s5_winrate = win_rate
                    st.session_state.s5_abandon = abandon
                    with st.spinner("AI综合批改..."):
                        grade = call_ai(STEP5_GRADER.format(
                            context=st.session_state.context,
                            direction=st.session_state.s1_dir,
                            confidence=st.session_state.s1_confidence,
                            market_state=st.session_state.s2_state,
                            ev_judgment=st.session_state.s3_reason[:100],
                            signal=st.session_state.s4_signal[:80],
                            decision=decision, win_rate=win_rate, abandon=abandon
                        ), max_tokens=480)
                    st.session_state.grade_s5 = grade
                    parse_grade_and_score(grade, "Trade Plan")
                    st.session_state.phase = "five_grade"
                    st.rerun()
                else:
                    st.warning("请填写放弃条件")

    # ========== 五步汇总 ==========
    elif phase == "five_grade":
        st.markdown("""
        <div class="step-header" style="border-left-color:#22c55e;background:#f0fdf4;">
            <div class="step-num" style="background:#22c55e;">✓</div>
            <div><div class="step-title">决策完成</div><div class="step-desc">进入市场验证</div></div>
        </div>""", unsafe_allow_html=True)

        st.plotly_chart(build_chart(df, obs_end, sh=sh, sl=sl), use_container_width=True)

        if st.session_state.grade_s5:
            render_ai_box(st.session_state.grade_s5)

        col1, col2 = st.columns(2)
        with col1:
            if st.button("进入验证", type="primary", use_container_width=True):
                st.session_state.phase = "verify"
                st.rerun()
        with col2:
            if st.button("跳过验证", use_container_width=True):
                elapsed = time.time() - (st.session_state.round_start or time.time())
                st.session_state.round_times.append(elapsed)
                st.session_state.phase = "complete"
                st.rerun()

    # ========== 验证 ==========
    elif phase == "verify":
        ver_total = st.session_state.ver_total
        display_bars = min(ver_total, len(df) - obs_end)

        st.markdown("""
        <div class="step-header" style="border-left-color:#7c3aed;background:#f5f3ff;">
            <div class="step-num" style="background:#7c3aed;">V</div>
            <div><div class="step-title">市场验证</div><div class="step-desc">假设存活了多久？</div></div>
        </div>""", unsafe_allow_html=True)

        st.plotly_chart(build_chart(df, obs_end, extra_bars=display_bars, sh=sh, sl=sl), use_container_width=True)

        new_bars = get_new_bars_text(df, obs_end, display_bars)
        st.info(f"**验证区 K线:**\n{new_bars}")

        st.markdown(f"""
        <div style="background:#fafafa;border:1px solid #e8e8e8;border-radius:6px;padding:10px;margin:8px 0;">
            <div style="font-size:11px;color:#888;">你的交易假设</div>
            <div>{st.session_state.s1_dir} · {st.session_state.s1_confidence} · {st.session_state.s5_decision}</div>
            <div style="font-size:12px;color:#666;">放弃条件: {st.session_state.s5_abandon[:80]}</div>
        </div>""", unsafe_allow_html=True)

        with st.form("ver_form"):
            ver_choice = st.radio("假设状态:", VER_OPTIONS)
            ver_reason = st.text_area("理由:", height=80)
            if st.form_submit_button("提交", type="primary"):
                if ver_reason:
                    with st.spinner("AI分析中..."):
                        grade = call_ai(VERIFICATION_GRADER.format(
                            direction=st.session_state.s1_dir,
                            confidence=st.session_state.s1_confidence,
                            abandon=st.session_state.s5_abandon,
                            new_bars=new_bars,
                            ver_choice=ver_choice,
                            ver_reason=ver_reason
                        ), max_tokens=370)
                    st.session_state.grade_ver = grade
                    parse_grade_and_score(grade, "Signal")
                    st.session_state.phase = "ver_result"
                    st.rerun()
                else:
                    st.warning("请填写理由")

    # ========== 验证结果 ==========
    elif phase == "ver_result":
        display_bars = min(st.session_state.ver_total, len(df) - obs_end)

        st.plotly_chart(build_chart(df, obs_end, extra_bars=display_bars, sh=sh, sl=sl), use_container_width=True)

        if st.session_state.grade_ver:
            render_ai_box(st.session_state.grade_ver, style="ok")

        # 客观评分（基于假设存活）
        score, desc = compute_verification_score(
            df, obs_end, st.session_state.ver_total,
            st.session_state.s1_dir, st.session_state.s1_confidence
        )
        if score is not None:
            st.session_state.verification_scores.append(score)
            surv_style = "ai-box-ok" if score >= 0.7 else ("ai-box-warn" if score >= 0.4 else "ai-box")
            surv_label = "✅ 假设存活验证" if score >= 0.7 else ("⚠️ 假设存活验证" if score >= 0.4 else "❌ 假设存活验证")
            st.markdown(f"""
            <div class="{surv_style}">
                <div class="ai-label">{surv_label}</div>
                <div class="ai-text">{desc}</div>
            </div>""", unsafe_allow_html=True)

        col1, col2 = st.columns(2)
        with col1:
            if st.button("完成,下一轮", type="primary", use_container_width=True):
                elapsed = time.time() - (st.session_state.round_start or time.time())
                st.session_state.round_times.append(elapsed)
                st.session_state.phase = "complete"
                st.rerun()
        with col2:
            if st.button("出报告", use_container_width=True):
                st.session_state.phase = "report"
                st.rerun()

    # ========== 完成 ==========
    elif phase == "complete":
        st.markdown(f"""
        <div class="step-header" style="border-left-color:#22c55e;background:#f0fdf4;">
            <div class="step-num" style="background:#22c55e;">✓</div>
            <div><div class="step-title">第{st.session_state.round_num}轮完成</div></div>
        </div>""", unsafe_allow_html=True)

        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("同品种下一轮", type="primary", use_container_width=True):
                if start_round(df):
                    st.session_state.phase = "step1"
                st.rerun()
        with col2:
            if st.button("换品种", use_container_width=True):
                code = random.choice([c for codes in EXCHANGES.values() for c in codes])
                if load_sym(code, period):
                    if start_round(st.session_state.df):
                        st.session_state.phase = "step1"
                st.rerun()
        with col3:
            if st.button("出报告", use_container_width=True):
                st.session_state.phase = "report"
                st.rerun()

    # ========== 报告 ==========
    elif phase == "report":
        st.markdown("## 📊 训练报告")

        rounds = st.session_state.round_num
        times = st.session_state.round_times
        avg_t = np.mean(times) if times else 0
        total_min = (time.time() - st.session_state.session_start) / 60 if st.session_state.session_start else 0

        accs = {s: get_step_accuracy(s) for s in FIVE_STEPS}
        surv = np.mean(st.session_state.verification_scores) * 100 if st.session_state.verification_scores else 0

        st.markdown('<div class="score-grid">', unsafe_allow_html=True)
        for step in FIVE_STEPS:
            acc = accs[step]
            color = "#22c55e" if acc >= 70 else ("#f97316" if acc >= 40 else "#ef4444")
            st.markdown(f'<div class="score-card"><div class="score-val" style="color:{color};">{acc}%</div><div class="score-lbl">{step}</div></div>', unsafe_allow_html=True)
        surv_color = "#22c55e" if surv >= 70 else ("#f97316" if surv >= 40 else "#ef4444")
        st.markdown(f'<div class="score-card"><div class="score-val" style="color:{surv_color};">{surv:.0f}%</div><div class="score-lbl">假设存活率</div></div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown(f"""
        <div style="display:flex;gap:12px;margin:12px 0;">
            <div class="score-card"><div class="score-val">{rounds}</div><div class="score-lbl">训练轮次</div></div>
            <div class="score-card"><div class="score-val">{avg_t:.0f}s</div><div class="score-lbl">平均用时</div></div>
            <div class="score-card"><div class="score-val">{total_min:.0f}m</div><div class="score-lbl">总时长</div></div>
        </div>""", unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("### 🤖 教练报告")

        if st.session_state.coach_report is None:
            with st.spinner("生成中..."):
                report = call_ai(COACH_REPORT_PROMPT.format(
                    symbol=symbol or "未知", period=PERIODS.get(period, ""), rounds=rounds, total_min=total_min,
                    s1=accs["Always In"], s2=accs["Market State"], s3=accs["Expected Value"],
                    s4=accs["Signal"], s5=accs["Trade Plan"], survival=surv,
                    errors="\n".join(st.session_state.round_errors[-5:]) or "无记录"
                ), max_tokens=600, temp=0.5)
            st.session_state.coach_report = report

        st.markdown(f"""
        <div style="background:#f0f4ff;border:1px solid #93c5fd;border-radius:10px;padding:24px;">
            <div style="font-size:11px;color:#1a56db;">AL BROOKS 教练点评</div>
            <div style="color:#222;line-height:1.9;">{st.session_state.coach_report}</div>
        </div>""", unsafe_allow_html=True)

        col1, col2 = st.columns(2)
        with col1:
            if st.button("重新开始", type="primary", use_container_width=True):
                for k in list(st.session_state.keys()): del st.session_state[k]
                st.rerun()
        with col2:
            if st.button("换品种继续", use_container_width=True):
                code = random.choice([c for codes in EXCHANGES.values() for c in codes])
                if load_sym(code, period):
                    if start_round(st.session_state.df):
                        st.session_state.coach_report = None
                        st.session_state.phase = "step1"
                        st.session_state.practice_count += 1
                        st.session_state.session_start = time.time()
                st.rerun()


if __name__ == "__main__":
    main()
