"""
Al Brooks 日内机会训练器 V3.0
融合版：V12.1的交互模式 + 结构预处理 + AI精准批改

核心改进：
1. 结构预处理模块：程序精确计算高低点序列、通道、特殊K线
2. AI三层角色：批改者（精准）/ 讲师（答错触发）/ 教练（结束报告）
3. 保留V12.1的主动观察流程：三问题 → AI批改 → 验证

训练流程：
观察30根K线 → 三问题（行为/依据/反证）→ AI用结构数据精准批改
→ 推进新K线逐根验证 → AI更新点评 → 结束报告
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

# Brooks知识库
BROOKS_KB = {
    "上升趋势": "上升趋势核心是HH+HL序列。强趋势中每次回调都是买机会，回调浅且快速被买回是强势信号。放弃条件：出现LH或LL。",
    "下降趋势": "下降趋势核心是LH+LL序列。强趋势中每次反弹都受压，反弹浅且快速被卖出是强势信号。放弃条件：出现HL或HH。",
    "横盘震荡": "横盘中70%的突破会失败。买卖双方势均力敌，高卖低买为主。等突破后回测确认，不追突破。",
    "上升通道": "接近上轨是卖方活跃区，接近下轨是低风险买入区。突破上轨向上可能是末期竭尽，要警惕反转。",
    "下降通道": "接近下轨买方尝试介入，接近上轨卖方压力大。突破上轨才是真正反转信号。",
    "楔形": "楔形是三推形态，每推动量衰减。需等趋势线有效突破才确认，不在楔形末端盲目逆势。",
    "双顶": "两次测试高点均未突破。第二顶更弱（影线长、实体小）时更可靠。有效跌破颈线是入场点。",
    "双底": "两次测试低点均获支撑。第二底更强（影线短、收盘高）时更可靠。有效突破颈线是入场点。",
    "三角形": "高低点收敛，通常延续前趋势（60%）。突破在顶点前2/3完成，超过顶点才突破失败率更高。",
    "内包K线": "前根K线范围内完成博弈，能量积蓄。连续内包后突破更有力。在趋势中内包后突破方向通常与趋势一致。",
    "外包K线": "同时测试两方向，看收盘位置定方向。收盘偏高看多，收盘偏低看空，收盘中间方向不明。",
    "竭尽形态": "大K线后出现反向K线，说明极端买/卖方能量被吸收。不等于立即反转，需等结构确认。",
}

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
    """纯numpy实现摆动点识别，无需scipy"""
    high = df['high'].values
    low  = df['low'].values
    n = len(high)
    sh, sl = [], []
    for i in range(order, n - order):
        h_win = high[i-order : i+order+1]
        l_win = low[i-order  : i+order+1]
        if high[i] >= max(h_win):
            sh.append((i, float(high[i])))
        if low[i] <= min(l_win):
            sl.append((i, float(low[i])))
    # 去重：相邻重复高/低点只保留最极端的那个
    def dedup(pts, key_fn, cmp_fn):
        if not pts: return pts
        result = [pts[0]]
        for p in pts[1:]:
            if p[0] - result[-1][0] <= order:
                if cmp_fn(p[1], result[-1][1]):
                    result[-1] = p
            else:
                result.append(p)
        return result
    sh = dedup(sh, lambda x: x[1], lambda a, b: a > b)
    sl = dedup(sl, lambda x: x[1], lambda a, b: a < b)
    return sh, sl


def classify_trend(sh, sl):
    if len(sh) < 2 or len(sl) < 2: return "数据不足"
    rh = [v for _, v in sh[-3:]]; rl = [v for _, v in sl[-3:]]
    hh = all(rh[i] > rh[i-1] for i in range(1, len(rh)))
    hl = all(rl[i] > rl[i-1] for i in range(1, len(rl)))
    lh = all(rh[i] < rh[i-1] for i in range(1, len(rh)))
    ll = all(rl[i] < rl[i-1] for i in range(1, len(rl)))
    if hh and hl: return "上升趋势"
    elif lh and ll: return "下降趋势"
    elif lh and hl: return "收敛三角形（震荡）"
    elif hh and ll: return "扩张震荡"
    else: return "横盘震荡"


def detect_channel(sh, sl):
    if len(sh) < 2 or len(sl) < 2: return "无明确通道"
    hx = [p[0] for p in sh[-3:]]; hy = [p[1] for p in sh[-3:]]
    lx = [p[0] for p in sl[-3:]]; ly = [p[1] for p in sl[-3:]]
    hs = np.polyfit(hx, hy, 1)[0] if len(hx) >= 2 else 0
    ls = np.polyfit(lx, ly, 1)[0] if len(lx) >= 2 else 0
    if hs > 0 and ls > 0: return "上升通道"
    elif hs < 0 and ls < 0: return "下降通道"
    elif hs < 0 and ls > 0: return "收敛三角形"
    elif hs > 0 and ls < 0: return "扩张震荡区间"
    elif hs > 0 and ls >= 0 and hs > ls: return "上升楔形（动量衰减）"
    elif ls < 0 and hs <= 0 and ls < hs: return "下降楔形（动量衰减）"
    return "横盘区间"


def detect_doubles(sh, sl, tol=0.004):
    results = []
    if len(sh) >= 2:
        h1, h2 = sh[-2][1], sh[-1][1]
        if abs(h1 - h2) / max(h1, h2) < tol:
            quality = "第二顶更弱（更可靠）" if h2 <= h1 else "第二顶更强（谨慎）"
            results.append(f"双顶——{quality}")
    if len(sl) >= 2:
        l1, l2 = sl[-2][1], sl[-1][1]
        if abs(l1 - l2) / max(l1, l2) < tol:
            quality = "第二底更强（更可靠）" if l2 >= l1 else "第二底更弱（谨慎）"
            results.append(f"双底——{quality}")
    return results


def detect_specials(df, n=6):
    recent = df.tail(n).reset_index(drop=True)
    results = []
    # 连续内包
    streak = sum(1 for _ in iter(lambda: recent['is_inside'].iloc[::-1].cumprod().sum(), 0))
    streak = 0
    for v in reversed(recent['is_inside'].tolist()):
        if v: streak += 1
        else: break
    if streak >= 2: results.append(f"连续{streak}根内包（能量积蓄，突破待定）")
    # IOI
    bars = list(recent.itertuples())
    for i in range(len(bars)-2):
        if bars[i].is_inside and bars[i+1].is_outside and bars[i+2].is_inside:
            results.append("IOI形态（内-外-内，高概率突破前兆）"); break
    # 外包
    if recent.iloc[-1]['is_outside']:
        pos = "收盘偏高（多头）" if recent.iloc[-1]['close'] > recent.iloc[-1]['open'] else "收盘偏低（空头）"
        results.append(f"最新K线外包——{pos}")
    # 大K线
    if recent.iloc[-1]['is_big']:
        d = "多头" if recent.iloc[-1]['direction'] == 'bull' else "空头"
        results.append(f"大{d}K线（强势，注意跟进或竭尽）")
    # 竭尽
    if len(recent) >= 2:
        p, c = recent.iloc[-2], recent.iloc[-1]
        if p['is_big'] and c['direction'] != p['direction'] and c['body_ratio'] > 0.5:
            results.append("大K线后反向强K线——可能竭尽形态")
    return results


def get_position(df, sh, sl):
    if not sh or not sl: return "位置无法判断"
    price = df['close'].iloc[-1]
    nh, nl = sh[-1][1], sl[-1][1]
    rng = nh - nl
    parts = []
    if rng > 0:
        r = (price - nl) / rng
        if r > 0.85: parts.append(f"区间上轨附近（{r:.0%}），压力区")
        elif r < 0.15: parts.append(f"区间下轨附近（{r:.0%}），支撑区")
        else: parts.append(f"区间中部（{r:.0%}）")
    if sh and abs(price - sh[-1][1]) / sh[-1][1] < 0.003:
        parts.append("极度接近前高（突破or双顶关键节点）")
    if sl and abs(price - sl[-1][1]) / sl[-1][1] < 0.003:
        parts.append("极度接近前低（突破or双底关键节点）")
    return "；".join(parts) if parts else "位置中性"


def build_structure(df, n_bars=60):
    """
    核心：将K线数据转化为AI可读的结构描述
    这是AI批改精准的关键
    """
    if len(df) < 15: return None, None
    data = df.tail(n_bars).copy().reset_index(drop=True)
    data = compute_bar_features(data)
    order = max(2, len(data) // 15)
    sh, sl = find_swings(data, order)
    trend = classify_trend(sh, sl)
    channel = detect_channel(sh, sl)
    doubles = detect_doubles(sh, sl)
    specials = detect_specials(data)
    position = get_position(data, sh, sl)

    ema20 = data['close'].ewm(span=20).mean().iloc[-1]
    price = data['close'].iloc[-1]
    ema_desc = f"{'上' if price > ema20 else '下'}方，偏离{abs(price-ema20)/ema20*100:.1f}%"

    # 高低点序列文字
    sh_vals = "→".join(f"{v:.0f}" for _, v in sh[-4:]) if len(sh) >= 2 else "不足"
    sl_vals = "→".join(f"{v:.0f}" for _, v in sl[-4:]) if len(sl) >= 2 else "不足"
    sh_dir = ("HH逐步抬高" if all(sh[i][1] > sh[i-1][1] for i in range(max(1,len(sh)-3),len(sh)))
              else "LH逐步降低" if all(sh[i][1] < sh[i-1][1] for i in range(max(1,len(sh)-3),len(sh)))
              else "混合") if len(sh) >= 2 else "不足"
    sl_dir = ("HL逐步抬高" if all(sl[i][1] > sl[i-1][1] for i in range(max(1,len(sl)-3),len(sl)))
              else "LL逐步降低" if all(sl[i][1] < sl[i-1][1] for i in range(max(1,len(sl)-3),len(sl)))
              else "混合") if len(sl) >= 2 else "不足"

    # 最近8根K线特征
    recent8 = data.tail(8)
    bar_desc = []
    for i, row in recent8.iterrows():
        d = "阳" if row['direction'] == 'bull' else "阴"
        sz = "大" if row['is_big'] else ("内包" if row['is_inside'] else ("外包" if row['is_outside'] else "普通"))
        bar_desc.append(f"K{i+1}:{sz}{d}({row['body_ratio']:.0%}实体,↑{row['upper_ratio']:.0%}↓{row['lower_ratio']:.0%})")

    summary = f"""【程序精确计算的结构数据——AI批改必须基于此，不得凭感觉】
趋势：{trend}
高点序列：{sh_vals}（{sh_dir}）
低点序列：{sl_vals}（{sl_dir}）
通道/形态：{channel}
双顶/双底：{"；".join(doubles) if doubles else "无"}
特殊K线：{"；".join(specials) if specials else "无"}
当前位置：{position}
EMA20关系：价格在EMA20{ema_desc}
最近8根K线：{" | ".join(bar_desc)}"""

    return summary, {
        "trend": trend, "channel": channel, "doubles": doubles,
        "specials": specials, "position": position,
        "swing_highs": sh, "swing_lows": sl
    }


# ==================== AI提示词 ====================

# 批改者：精准指出对错，有结构数据撑腰
GRADER_PROMPT = """你是Al Brooks价格行为教练，正在批改学员的读图练习。

{structure}

【学员回答】
① 市场在做什么：{observation}
② 判断依据（K线引用）：{evidence}  
③ 反证信号：{fail_signal}

【批改标准——必须全部覆盖】
1. 趋势判断：学员说的趋势是否与程序计算的结构一致？引用高低点序列说明
2. K线引用质量：学员引用的K线编号是否指向真正关键的K线？有没有遗漏更重要的K线？
3. 反证信号设置：反证信号是否合理？是否与当前结构匹配？
4. 最重要的一个遗漏或误判（如果有）

【输出格式】
先给出总体评价（1句话：读对了哪里，最大的问题在哪里）
然后逐条批改（引用结构数据，不允许泛泛而谈）
最后：如果你是Al Brooks，你看这段走势第一眼会注意到什么？

控制在220字以内。语气：直接，像经验丰富的老师，不废话。"""


# 讲师：Brooks语言重新讲解，结合当前图
TEACHER_PROMPT = """你是Al Brooks，用你的风格讲解学员理解偏差的地方。

{structure}

【学员的主要误判】
{main_error}

【当前图表的关键结构】
趋势：{trend}，通道：{channel}

【Brooks知识库参考】
{knowledge}

用第一人称，结合这张图的高低点序列、K线特征具体讲解。
告诉学员：遇到这种结构，第一步看什么，第二步验证什么。
不要背教材，要结合数据说话。
控制在200字。"""


# 验证点评：新K线出现后，精准分析对剧本的影响
VERIFICATION_PROMPT = """你是Al Brooks价格行为教练。

{structure}

【学员之前的判断】
市场行为：{observation}
反证信号：{fail_signal}

【新出现的K线】
{new_bars}

【学员的验证说明】
{user_verification}

【点评要求】
1. 新K线是否触发了学员设定的反证信号？
2. 新K线对当前结构（趋势/通道）的影响是什么？
3. 根据结构数据，学员的判断现在应该维持/修正/放弃？给出明确建议
4. 下一根K线出现时，最值得观察的是什么行为？

控制在180字。直接给结论，不绕弯子。"""


# 教练：结束后薄弱点报告
COACH_PROMPT = """你是Al Brooks训练营首席教练，给学员写本次训练报告。

【训练数据】
品种：{symbol} {period}，完成{rounds}轮
平均用时：{avg_time:.0f}秒/轮

【各轮主要问题】
{round_errors}

【报告要求】
1. 最薄弱的1-2个能力（从：趋势判断 / K线引用精度 / 反证信号设置 / 验证准确性 中选）
   说明为什么，引用具体错误
2. 下次训练的可操作建议（具体到"练什么"，不要说"多练习"）
3. 本次做得最好的一点
4. 一句真诚的鼓励（不要鸡汤，要具体到这个人的情况）

控制在280字。"""


def get_client():
    key = st.secrets.get("OPENAI_API_KEY", "")
    url = st.secrets.get("OPENAI_BASE_URL", "https://api.deepseek.com")
    model = st.secrets.get("OPENAI_MODEL", "deepseek-chat")
    return OpenAI(base_url=url, api_key=key), model, bool(key)


def call_ai(prompt, max_tokens=450, temp=0.3):
    client, model, has_key = get_client()
    if not has_key:
        return "（未配置API Key，请在Streamlit secrets中设置OPENAI_API_KEY）"
    try:
        r = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temp, max_tokens=max_tokens
        )
        return r.choices[0].message.content
    except Exception as e:
        return f"AI调用失败：{str(e)[:100]}"


def ai_grade(structure, observation, evidence, fail_signal):
    return call_ai(GRADER_PROMPT.format(
        structure=structure, observation=observation,
        evidence=evidence, fail_signal=fail_signal or "未设置"
    ), max_tokens=500)


def ai_teach(structure, main_error, trend, channel, knowledge_key):
    kb = BROOKS_KB.get(knowledge_key, BROOKS_KB["横盘震荡"])
    return call_ai(TEACHER_PROMPT.format(
        structure=structure, main_error=main_error,
        trend=trend, channel=channel, knowledge=kb
    ), max_tokens=450, temp=0.5)


def ai_verify(structure, observation, fail_signal, new_bars, user_verification):
    return call_ai(VERIFICATION_PROMPT.format(
        structure=structure, observation=observation,
        fail_signal=fail_signal or "未设置",
        new_bars=new_bars, user_verification=user_verification
    ), max_tokens=400)


def ai_coach(symbol, period, rounds, avg_time, round_errors):
    errors_text = "\n".join(f"第{i+1}轮：{e}" for i, e in enumerate(round_errors)) if round_errors else "无记录"
    return call_ai(COACH_PROMPT.format(
        symbol=symbol, period=period, rounds=rounds,
        avg_time=avg_time, round_errors=errors_text
    ), max_tokens=550, temp=0.5)


# ==================== 数据加载 ====================

@st.cache_data(ttl=1800, show_spinner=False)
def load_data(symbol, period):
    try:
        df = ak.futures_zh_minute_sina(symbol=f"{symbol}0", period=period)
        if df is None or len(df) < 50: return None
        return df.rename(columns={
            "date": "time", "open": "open", "high": "high",
            "low": "low", "close": "close", "volume": "volume"
        }).reset_index(drop=True)
    except: return None


# ==================== 图表 ====================

def build_chart(df, obs_end, extra_bars=0, sh=None, sl=None):
    """
    obs_end: 观察窗口结束的绝对index
    extra_bars: 验证阶段额外显示的K线数
    """
    view_end = obs_end + extra_bars
    display_n = 80
    start = max(0, view_end - display_n)
    data = df.iloc[start:view_end].copy().reset_index(drop=True)
    n = len(data)
    offset = start  # 绝对index偏移

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        vertical_spacing=0.02, row_heights=[0.76, 0.24])

    # K线
    fig.add_trace(go.Candlestick(
        x=data.index, open=data['open'], high=data['high'],
        low=data['low'], close=data['close'], showlegend=False,
        increasing_line_color="#ef5350", decreasing_line_color="#26a69a",
    ), row=1, col=1)

    # EMA20
    ema = data['close'].ewm(span=20).mean()
    fig.add_trace(go.Scatter(
        x=data.index, y=ema,
        line=dict(color="#ff9800", width=1.2, dash="dot"),
        name="EMA20", showlegend=True
    ), row=1, col=1)

    # 摆动点标注
    if sh:
        for idx, val in sh:
            pi = idx - offset
            if 0 <= pi < n:
                fig.add_annotation(x=pi, y=val, text="H",
                    font=dict(size=9, color="#ef5350", family="monospace"),
                    showarrow=False, yshift=12, row=1, col=1)
    if sl:
        for idx, val in sl:
            pi = idx - offset
            if 0 <= pi < n:
                fig.add_annotation(x=pi, y=val, text="L",
                    font=dict(size=9, color="#26a69a", family="monospace"),
                    showarrow=False, yshift=-14, row=1, col=1)

    # 观察窗口边界线（橙色虚线）
    obs_in_chart = obs_end - start - 1
    if 0 <= obs_in_chart < n:
        fig.add_vline(x=obs_in_chart, line_dash="dash",
                      line_color="#ff9800", line_width=1.5, opacity=0.8)
        fig.add_annotation(x=obs_in_chart, y=data['high'].max(),
            text="观察截止", font=dict(size=9, color="#ff9800"),
            showarrow=False, yshift=14)

    # 验证区高亮
    if extra_bars > 0:
        fig.add_vrect(x0=obs_in_chart, x1=n-1,
                      fillcolor="#388bfd", opacity=0.07,
                      layer="below", line_width=0)

    # K线编号（每5根标一次）
    for i in range(4, n, 5):
        row_data = data.iloc[i]
        abs_num = i + offset + 1
        y = row_data['low'] if row_data['close'] >= row_data['open'] else row_data['high']
        shift = -14 if row_data['close'] >= row_data['open'] else 14
        fig.add_annotation(x=i, y=y, text=f"K{abs_num}",
            showarrow=False, font=dict(size=8, color="#888888"), yshift=shift)

    # 成交量
    vc = ["#ef5350" if c >= o else "#26a69a" for o, c in zip(data['open'], data['close'])]
    fig.add_trace(go.Bar(x=data.index, y=data['volume'],
                         marker_color=vc, showlegend=False, opacity=0.5), row=2, col=1)

    fig.update_layout(
        xaxis_rangeslider_visible=False, height=520,
        margin=dict(l=8, r=8, t=28, b=8),
        paper_bgcolor="#ffffff", plot_bgcolor="#f8f9fa",
        font=dict(color="#333333"),
        legend=dict(orientation="h", yanchor="bottom", y=1.01,
                    xanchor="right", x=1, font=dict(size=11))
    )
    fig.update_xaxes(showgrid=True, gridcolor="#e8e8e8", gridwidth=0.5,
                     zeroline=False, showticklabels=False, row=1, col=1)
    fig.update_xaxes(showgrid=True, gridcolor="#e8e8e8", gridwidth=0.5, zeroline=False, row=2, col=1)
    fig.update_yaxes(showgrid=True, gridcolor="#e8e8e8", gridwidth=0.5, zeroline=False)
    return fig


# ==================== State管理 ====================

def init_state():
    d = {
        "df": None, "symbol": None, "period": "15",
        # 核心流程
        # phases: select → observe → grading → verify → ver_result → complete → report
        "phase": "select",
        "obs_end": 0,           # 观察窗口结束（绝对index）
        "ver_bar": 0,           # 当前验证到第几根（相对obs_end）
        "ver_max": 5,           # 每轮验证几根
        # 结构数据
        "structure": "",        # 文字描述（喂给AI）
        "struct_data": {},      # 字典（用于图表标注）
        # 用户输入
        "user_obs": "",
        "user_ev": "",
        "user_fail": "",
        "user_ver": "",
        # AI输出
        "ai_grade": None,
        "ai_teach": None,
        "ai_ver": None,
        # 训练统计
        "round_num": 0,
        "round_times": [],
        "round_errors": [],
        "round_start": None,
        "coach_report": None,
        "practice_count": 0,
        "session_start": None,
    }
    for k, v in d.items():
        if k not in st.session_state: st.session_state[k] = v


def reset_round():
    for k in ["user_obs", "user_ev", "user_fail", "user_ver",
              "ai_grade", "ai_teach", "ai_ver"]:
        st.session_state[k] = None if k.startswith("ai") else ""
    st.session_state.ver_bar = 0
    st.session_state.round_start = time.time()


def load_sym(code, period):
    with st.spinner(f"加载 {code} {PERIODS[period]}..."):
        df = load_data(code, period)
    if df is None or len(df) < 60: return False
    st.session_state.df = df
    st.session_state.symbol = code
    st.session_state.period = period
    return True


def start_round(df, n_bars=60):
    """随机选取观察窗口，计算结构"""
    max_offset = max(0, len(df) - n_bars - st.session_state.ver_max - 2)
    offset = random.randint(0, max_offset)
    obs_end = len(df) - offset
    st.session_state.obs_end = obs_end
    result = build_structure(df.iloc[:obs_end], n_bars)
    if result[0] is None: return False
    st.session_state.structure = result[0]
    st.session_state.struct_data = result[1]
    st.session_state.round_num += 1
    reset_round()
    return True


def get_new_bars_text(df, obs_end, n):
    """获取验证K线的文字描述"""
    lines = []
    for i in range(n):
        idx = obs_end + i
        if idx >= len(df): break
        row = df.iloc[idx]
        o, h, l, c = row['open'], row['high'], row['low'], row['close']
        body = abs(c - o); total = h - l
        br = body / total if total > 0 else 0
        d = "阳" if c >= o else "阴"
        # 内外包
        if idx > 0:
            prev = df.iloc[idx-1]
            special = "（内包）" if h <= prev['high'] and l >= prev['low'] else \
                      "（外包）" if h > prev['high'] and l < prev['low'] else ""
        else: special = ""
        lines.append(f"K{idx+1}{special}：{d}线，O={o:.0f} H={h:.0f} L={l:.0f} C={c:.0f}，实体{br:.0%}")
    return "\n".join(lines)


# ==================== CSS ====================
CSS = """
<style>
.stApp { background-color: #ffffff; }
[data-testid="stSidebar"] { background-color: #f5f5f5 !important; border-right: 1px solid #e0e0e0; }

.phase-bar {
    background: #f0f7ff;
    border-left: 3px solid #1976d2;
    border-radius: 0 6px 6px 0;
    padding: 12px 16px; margin-bottom: 16px;
}
.phase-title { font-size: 16px; font-weight: 700; color: #1565c0; }
.phase-hint { font-size: 12px; color: #666; margin-top: 3px; }

.ai-box {
    background: #f0f7ff;
    border: 1px solid #90caf9;
    border-radius: 8px;
    padding: 16px; margin: 12px 0;
}
.ai-teach-box {
    background: #fff8f0;
    border: 1px solid #ffb74d;
    border-left: 3px solid #f57c00;
    border-radius: 8px;
    padding: 16px; margin: 12px 0;
}
.ai-ver-box {
    background: #f0fff4;
    border: 1px solid #81c784;
    border-radius: 8px;
    padding: 16px; margin: 12px 0;
}
.ai-label { font-size: 11px; color: #888; margin-bottom: 8px; letter-spacing: 0.05em; }
.ai-content { color: #333; line-height: 1.8; font-size: 14px; }

.structure-box {
    background: #fafafa; border: 1px solid #e0e0e0;
    border-radius: 6px; padding: 12px;
    font-family: monospace; font-size: 11px;
    color: #555; line-height: 1.7;
    white-space: pre-wrap;
}

.stat-row { display: flex; gap: 10px; margin: 12px 0; }
.stat-card {
    flex: 1; background: #f5f5f5; border: 1px solid #e0e0e0;
    border-radius: 8px; padding: 12px; text-align: center;
}
.stat-val { font-size: 24px; font-weight: 700; color: #1976d2; font-family: monospace; }
.stat-lbl { font-size: 11px; color: #888; margin-top: 3px; }

.stButton > button {
    border-radius: 6px; border: 1px solid #ccc;
    background: #f0f0f0; color: #333; font-weight: 500;
    transition: all 0.15s;
}
.stButton > button:hover { background: #1976d2; border-color: #1976d2; color: #fff; }
.stTextArea textarea { background: #fafafa !important; color: #333 !important; border-color: #ddd !important; }

.round-tag {
    display: inline-block; background: #e3f2fd;
    border: 1px solid #90caf9; border-radius: 20px;
    padding: 2px 12px; font-size: 12px; color: #1565c0;
    font-family: monospace;
}
.brooks-quote {
    border-left: 2px solid #f57c00; padding: 8px 14px;
    color: #888; font-style: italic; font-size: 12px; margin: 12px 0;
}
div[data-testid="stExpander"] { background: #fafafa; border: 1px solid #e0e0e0; border-radius: 8px; }
.stRadio [data-testid="stMarkdownContainer"] p { color: #333 !important; }
</style>
"""


# ==================== 主界面 ====================

def main():
    st.set_page_config(page_title="Al Brooks 训练器 V3", layout="wide")
    st.markdown(CSS, unsafe_allow_html=True)
    init_state()

    df = st.session_state.df
    phase = st.session_state.phase
    symbol = st.session_state.symbol
    period = st.session_state.period
    struct_data = st.session_state.struct_data

    # ── 侧边栏 ──
    with st.sidebar:
        st.markdown("## Al Brooks 训练器 V3")
        st.caption("主动观察 · 结构批改 · 精准反馈")
        st.markdown("---")

        # 统计
        if st.session_state.round_num > 0:
            times = st.session_state.round_times
            avg_t = np.mean(times) if times else 0
            st.markdown(f"""
            <div class="stat-row">
                <div class="stat-card">
                    <div class="stat-val">{st.session_state.round_num}</div>
                    <div class="stat-lbl">训练轮次</div>
                </div>
                <div class="stat-card">
                    <div class="stat-val">{avg_t:.0f}s</div>
                    <div class="stat-lbl">平均用时</div>
                </div>
            </div>""", unsafe_allow_html=True)

        st.markdown("---")
        st.caption("周期 & 品种")
        period_sel = st.selectbox("K线周期", list(PERIODS.keys()),
                                  format_func=lambda x: PERIODS[x], index=1)

        for cat, codes in EXCHANGES.items():
            with st.expander(cat, expanded=False):
                cols = st.columns(2)
                for i, code in enumerate(codes):
                    if cols[i % 2].button(code, key=f"s_{code}",
                                          use_container_width=True,
                                          help=SYMBOL_NAMES.get(code, "")):
                        if load_sym(code, period_sel):
                            if start_round(st.session_state.df):
                                st.session_state.phase = "observe"
                                st.session_state.practice_count += 1
                                st.session_state.session_start = time.time()
                            st.rerun()
                        else:
                            st.error(f"{code} 加载失败")

        st.markdown("---")
        if st.button("🎲 随机品种", use_container_width=True):
            code = random.choice([c for codes in EXCHANGES.values() for c in codes])
            if load_sym(code, period_sel):
                if start_round(st.session_state.df):
                    st.session_state.phase = "observe"
                    st.session_state.practice_count += 1
                    st.session_state.session_start = time.time()
            st.rerun()

        if phase not in ("select", "report"):
            st.markdown("---")
            if st.button("📊 结束出报告", use_container_width=True):
                st.session_state.phase = "report"
                st.rerun()
            if st.button("🔄 重置", use_container_width=True):
                for k in list(st.session_state.keys()):
                    del st.session_state[k]
                st.rerun()

        st.markdown("""
        <div class="brooks-quote">
        "The market is always in some kind of a trend or trading range. Your job is to figure out which one."<br>— Al Brooks
        </div>""", unsafe_allow_html=True)

    # ── 欢迎页 ──
    if phase == "select":
        st.markdown("## 👈 从左侧选择品种开始")
        st.markdown("""
        <div style="background:#f0f7ff;border:1px solid #90caf9;border-radius:10px;padding:24px;margin-top:16px;">
        <h3 style="color:#1565c0;margin-top:0;">Al Brooks 日内机会训练器 V3.0</h3>
        <p style="color:#666;">主动观察 + 程序结构验证 + AI精准批改，三合一</p>
        <hr style="border-color:#e0e0e0;">
        <h4>每轮训练流程</h4>
        <ol style="color:#333;line-height:2.2;">
          <li><b>观察30根K线</b> — 回答三个问题：市场在做什么？依据哪些K线？反证信号是什么？</li>
          <li><b>AI精准批改</b> — 程序计算HH/HL/通道/特殊K线，AI用真实数据批改你的观察</li>
          <li><b>Brooks讲解</b>（可选）— 答错或存疑时，AI用Brooks语言结合当前图重新讲解</li>
          <li><b>逐根验证</b> — 推进5根新K线，每根验证你的判断是否仍然成立</li>
          <li><b>AI验证点评</b> — 新K线对剧本的影响，概率如何更新，下根K线看什么</li>
          <li><b>教练报告</b> — 训练结束后，AI指出薄弱维度 + 可操作建议</li>
        </ol>
        <p style="color:#888;font-size:13px;">
        ✦ AI批改基于程序精确计算的结构数据，不是凭感觉——高低点序列、通道斜率、内外包形态，全部数字化。
        </p>
        </div>""", unsafe_allow_html=True)
        return

    if df is None:
        st.warning("请先选择品种"); return

    # 公共顶栏
    trend = struct_data.get("trend", "")
    channel = struct_data.get("channel", "")
    c1, c2, c3, c4 = st.columns([2, 1.2, 1.2, 0.8])
    with c1: st.markdown(f"**{symbol}** · {SYMBOL_NAMES.get(symbol,'')} · {PERIODS[period]}")
    with c2:
        tc = "#26a69a" if "上升" in trend else ("#ef5350" if "下降" in trend else "#f57c00")
        st.markdown(f'<span style="color:{tc};font-weight:600;">{trend}</span>', unsafe_allow_html=True)
    with c3: st.markdown(f'<span style="color:#888;font-size:13px;">{channel}</span>', unsafe_allow_html=True)
    with c4: st.markdown(f'<span class="round-tag">第{st.session_state.round_num}轮</span>', unsafe_allow_html=True)

    sh = struct_data.get("swing_highs", [])
    sl = struct_data.get("swing_lows", [])
    obs_end = st.session_state.obs_end

    # ── 观察阶段 ──
    if phase == "observe":
        st.markdown("""
        <div class="phase-bar">
          <div class="phase-title">Step 1 · 观察并回答三个问题</div>
          <div class="phase-hint">只描述你看到的行为，引用具体K线编号，不要先给结论</div>
        </div>""", unsafe_allow_html=True)

        st.plotly_chart(build_chart(df, obs_end, sh=sh, sl=sl), use_container_width=True)

        with st.form("obs_form"):
            obs = st.text_area("① 市场在做什么？（描述行为，不贴标签）",
                placeholder="例如：K20到K30连续出现阴线，每根收盘都在当根低位；K31出现阳线但未能收复K28的高点...",
                height=90)
            ev = st.text_area("② 你的判断依据是哪些K线？（必须引用编号）",
                placeholder="例如：K25是关键——它的收盘直接跌破了K18的低点；K29反弹到K22附近受压...",
                height=90)
            fail = st.text_area("③ 什么情况下你会改变判断？（反证信号）",
                placeholder="例如：如果接下来出现一根实体大阳线，收盘超过K25的高点，我会重新评估空头判断...",
                height=70)
            submitted = st.form_submit_button("提交观察 → AI批改", type="primary")
            if submitted:
                if obs and ev:
                    st.session_state.user_obs = obs
                    st.session_state.user_ev = ev
                    st.session_state.user_fail = fail
                    with st.spinner("AI正在用结构数据批改..."):
                        grade = ai_grade(st.session_state.structure, obs, ev, fail)
                    st.session_state.ai_grade = grade
                    st.session_state.phase = "grading"
                    st.rerun()
                else:
                    st.warning("请至少填写前两个问题")

    # ── AI批改阶段 ──
    elif phase == "grading":
        st.markdown("""
        <div class="phase-bar">
          <div class="phase-title">Step 2 · AI批改结果</div>
          <div class="phase-hint">基于程序精确计算的结构数据，不是凭感觉</div>
        </div>""", unsafe_allow_html=True)

        st.plotly_chart(build_chart(df, obs_end, sh=sh, sl=sl), use_container_width=True)

        # 显示用户答案
        with st.expander("📋 你的观察", expanded=False):
            st.markdown(f"**市场行为：** {st.session_state.user_obs}")
            st.markdown(f"**判断依据：** {st.session_state.user_ev}")
            if st.session_state.user_fail:
                st.markdown(f"**反证信号：** {st.session_state.user_fail}")

        # AI批改
        st.markdown(f"""
        <div class="ai-box">
          <div class="ai-label">🎯 AI批改 · 结构数据支撑</div>
          <div class="ai-content">{st.session_state.ai_grade}</div>
        </div>""", unsafe_allow_html=True)

        # Brooks讲解按钮
        col1, col2, col3 = st.columns([1, 1, 1])
        with col1:
            if st.button("📖 Brooks详细讲解", use_container_width=True):
                if st.session_state.ai_teach is None:
                    with st.spinner("Brooks讲解中..."):
                        # 自动选择最相关的知识点
                        kk = trend
                        dbl = struct_data.get("doubles", [])
                        sp = struct_data.get("specials", [])
                        if dbl: kk = "双顶" if "双顶" in dbl[0] else "双底"
                        elif "楔形" in channel: kk = "楔形"
                        elif "三角" in channel: kk = "三角形"
                        elif sp and "内包" in sp[0]: kk = "内包K线"
                        elif sp and "外包" in sp[0]: kk = "外包K线"
                        elif sp and "竭尽" in "".join(sp): kk = "竭尽形态"
                        st.session_state.ai_teach = ai_teach(
                            st.session_state.structure,
                            f"学员观察：{st.session_state.user_obs[:100]}",
                            trend, channel, kk
                        )
                    st.rerun()
        with col2:
            if st.button("✏️ 重新观察", use_container_width=True):
                st.session_state.ai_grade = None
                st.session_state.ai_teach = None
                st.session_state.phase = "observe"
                st.rerun()
        with col3:
            if st.button("➡️ 进入验证", type="primary", use_container_width=True):
                st.session_state.ver_bar = 0
                st.session_state.phase = "verify"
                st.rerun()

        # Brooks讲解（如果有）
        if st.session_state.ai_teach:
            st.markdown(f"""
            <div class="ai-teach-box">
              <div class="ai-label">📖 Brooks讲解 · 结合当前图表</div>
              <div class="ai-content">{st.session_state.ai_teach}</div>
            </div>""", unsafe_allow_html=True)

        # 结构数据展示
        with st.expander("📊 程序计算的结构数据（AI批改依据）", expanded=False):
            st.markdown(f'<div class="structure-box">{st.session_state.structure}</div>',
                        unsafe_allow_html=True)

    # ── 验证阶段 ──
    elif phase == "verify":
        ver_bar = st.session_state.ver_bar
        ver_max = st.session_state.ver_max
        bar_num = ver_bar + 1

        # 检查是否超出数据范围
        if ver_bar >= ver_max or obs_end + ver_bar >= len(df):
            st.session_state.phase = "complete"
            st.rerun()

        st.markdown(f"""
        <div class="phase-bar">
          <div class="phase-title">Step 3 · 验证第 {bar_num}/{ver_max} 根新K线</div>
          <div class="phase-hint">预期：{st.session_state.user_fail or "（未设置反证信号）"}</div>
        </div>""", unsafe_allow_html=True)

        # 显示包含新K线的图表
        st.plotly_chart(build_chart(df, obs_end, extra_bars=bar_num, sh=sh, sl=sl),
                        use_container_width=True)

        # 新K线数据
        new_bar_text = get_new_bars_text(df, obs_end, bar_num)
        bar_lines = new_bar_text.strip().split("\n")
        latest_bar = bar_lines[-1] if bar_lines else ""
        st.info(f"**新K线：** {latest_bar}")

        with st.form(f"ver_form_{ver_bar}"):
            ver_input = st.text_area(
                f"这根K线出现后，你的判断如何？",
                placeholder="例如：仍然维持空头判断——这根阳线实体小、上影线长，说明反弹力度弱，没有触发我的反证信号...",
                height=90,
                key=f"ver_text_{ver_bar}"
            )
            submitted = st.form_submit_button("提交验证 → AI点评", type="primary")
            if submitted:
                if ver_input:
                    with st.spinner("AI分析新K线影响..."):
                        ver_grade = ai_verify(
                            st.session_state.structure,
                            st.session_state.user_obs,
                            st.session_state.user_fail,
                            new_bar_text, ver_input
                        )
                    st.session_state.ai_ver = ver_grade
                    st.session_state.user_ver = ver_input
                    st.session_state.phase = "ver_result"
                    st.rerun()
                else:
                    st.warning("请描述你的验证判断")

    # ── 验证反馈 ──
    elif phase == "ver_result":
        ver_bar = st.session_state.ver_bar
        ver_max = st.session_state.ver_max
        bar_num = ver_bar + 1

        st.markdown("""
        <div class="phase-bar">
          <div class="phase-title">Step 3 反馈 · 新K线点评</div>
        </div>""", unsafe_allow_html=True)

        st.plotly_chart(build_chart(df, obs_end, extra_bars=bar_num, sh=sh, sl=sl),
                        use_container_width=True)

        new_bar_text = get_new_bars_text(df, obs_end, bar_num)
        bar_lines = new_bar_text.strip().split("\n")
        st.info(f"**K线：** {bar_lines[-1] if bar_lines else ''}")

        st.markdown(f"""
        <div class="ai-ver-box">
          <div class="ai-label">✅ AI验证点评</div>
          <div class="ai-content">{st.session_state.ai_ver}</div>
        </div>""", unsafe_allow_html=True)

        col1, col2 = st.columns(2)
        with col1:
            next_bar = ver_bar + 1
            if next_bar < ver_max and obs_end + next_bar < len(df):
                if st.button(f"继续验证第{next_bar+1}根 →", type="primary", use_container_width=True):
                    st.session_state.ver_bar = next_bar
                    st.session_state.ai_ver = None
                    st.session_state.phase = "verify"
                    st.rerun()
            else:
                if st.button("完成验证 →", type="primary", use_container_width=True):
                    # 记录本轮
                    elapsed = time.time() - (st.session_state.round_start or time.time())
                    st.session_state.round_times.append(elapsed)
                    st.session_state.phase = "complete"
                    st.rerun()
        with col2:
            if st.button("跳过剩余验证", use_container_width=True):
                elapsed = time.time() - (st.session_state.round_start or time.time())
                st.session_state.round_times.append(elapsed)
                st.session_state.phase = "complete"
                st.rerun()

    # ── 本轮完成 ──
    elif phase == "complete":
        st.markdown(f"""
        <div class="phase-bar" style="border-left-color:#43a047;">
          <div class="phase-title" style="color:#2e7d32;">✅ 第{st.session_state.round_num}轮完成</div>
        </div>""", unsafe_allow_html=True)

        st.plotly_chart(build_chart(df, obs_end,
            extra_bars=min(st.session_state.ver_bar+1, st.session_state.ver_max),
            sh=sh, sl=sl), use_container_width=True)

        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("➡️ 同品种下一轮", type="primary", use_container_width=True):
                if start_round(df):
                    st.session_state.phase = "observe"
                st.rerun()
        with col2:
            if st.button("🎲 随机换品种", use_container_width=True):
                code = random.choice([c for codes in EXCHANGES.values() for c in codes
                                      if c != symbol])
                if load_sym(code, period):
                    if start_round(st.session_state.df):
                        st.session_state.phase = "observe"
                st.rerun()
        with col3:
            if st.button("📊 出训练报告", use_container_width=True):
                st.session_state.phase = "report"
                st.rerun()

    # ── 训练报告 ──
    elif phase == "report":
        st.markdown("## 📊 训练报告")

        rounds = st.session_state.round_num
        times = st.session_state.round_times
        avg_t = np.mean(times) if times else 0
        total_t = (time.time() - st.session_state.session_start) / 60 if st.session_state.session_start else 0

        st.markdown(f"""
        <div class="stat-row">
          <div class="stat-card"><div class="stat-val">{rounds}</div><div class="stat-lbl">训练轮次</div></div>
          <div class="stat-card"><div class="stat-val">{avg_t:.0f}s</div><div class="stat-lbl">平均用时/轮</div></div>
          <div class="stat-card"><div class="stat-val">{total_t:.0f}m</div><div class="stat-lbl">总训练时长</div></div>
          <div class="stat-card"><div class="stat-val">{st.session_state.practice_count}</div><div class="stat-lbl">累计训练次数</div></div>
        </div>""", unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("### 🤖 Al Brooks 教练报告")

        if st.session_state.coach_report is None:
            with st.spinner("教练报告生成中..."):
                report = ai_coach(
                    symbol or "未知", PERIODS.get(period, ""),
                    rounds, avg_t,
                    st.session_state.round_errors
                )
            st.session_state.coach_report = report

        st.markdown(f"""
        <div style="background:#f0f7ff;border:1px solid #90caf9;border-radius:10px;padding:24px;margin:12px 0;">
          <div style="font-size:11px;color:#1565c0;margin-bottom:12px;letter-spacing:0.05em;">AL BROOKS 教练点评</div>
          <div style="color:#333;line-height:1.9;font-size:14px;">{st.session_state.coach_report}</div>
        </div>""", unsafe_allow_html=True)

        st.markdown("---")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🔄 重新开始训练", type="primary", use_container_width=True):
                for k in list(st.session_state.keys()): del st.session_state[k]
                st.rerun()
        with col2:
            if st.button("🎲 换品种继续", use_container_width=True):
                code = random.choice([c for codes in EXCHANGES.values() for c in codes])
                if load_sym(code, period):
                    if start_round(st.session_state.df):
                        st.session_state.coach_report = None
                        st.session_state.phase = "observe"
                        st.session_state.practice_count += 1
                        st.session_state.session_start = time.time()
                st.rerun()


if __name__ == "__main__":
    main()
