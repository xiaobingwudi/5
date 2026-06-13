"""
Al Brooks 实盘逐K心智决策训练器 V5.0
核心：先判环境（State），再定方向（Always In）
验证：由AI终审裁决，而非死板的涨跌逻辑
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

# 五步标签（新顺序）
FIVE_STEPS = ["Market State", "Always In", "Location", "Signal", "Trade Plan"]

# Always In信心等级
CONFIDENCE_LEVELS = ["51%（勉强偏向）", "55%", "60%", "70%", "80%以上（强烈确信）"]

# 胜率类型
WIN_RATE_TYPES = ["高胜率 / 低盈亏比（如区间反转）",
                  "中等胜率 / 中等盈亏比（如趋势回调）",
                  "低胜率 / 高盈亏比（如突破交易）"]

# 验证选项
VER_OPTIONS = ["✅ 交易假设维持（逻辑仍然成立）",
               "⚠️ 交易假设减弱（仍持有但需观察）",
               "❌ 交易假设失效（需要退出或反向）"]

# Brooks经典信号选项
SIGNAL_OPTIONS = ["无明显信号（放弃）", "趋势K线（Trend Bar）", "反转棒（Reversal Bar）", "内包棒（Inside Bar）",
                  "H1 / H2（多头一买/二买）", "L1 / L2（空头一卖/二卖）", "ii（连续内包）", "IOI（内-外-内）"]

# ==================== 1. 强化版：结构预处理与特征提取 ====================

def compute_bar_features(df):
    """
    深度提取 Al Brooks 价格行为体系所需的微观K线特征
    """
    df = df.copy()
    # 确保列名小写
    df.columns = [c.lower() for c in df.columns]
    
    # 基础物理量
    df['body'] = abs(df['close'] - df['open'])
    df['total_range'] = df['high'] - df['low']
    df['body_ratio'] = np.where(df['total_range'] > 0, df['body'] / df['total_range'], 0)
    df['direction'] = np.where(df['close'] >= df['open'], 'bull', 'bear')
    
    # 引线计算
    df['upper_shadow'] = df['high'] - df[['open', 'close']].max(axis=1)
    df['lower_shadow'] = df[['open', 'close']].min(axis=1) - df['low']
    
    # 1. 核心：连续K线的重叠度（Overlap）- Brooks 区间市场的核心判定指标
    prev_high = df['high'].shift(1)
    prev_low = df['low'].shift(1)
    overlap_height = np.minimum(df['high'], prev_high) - np.maximum(df['low'], prev_low)
    df['is_overlap'] = overlap_height > 0
    
    # 2. 核心：收盘价在全根K线中的绝对相对位置
    df['close_pos_ratio'] = np.where(df['total_range'] > 0, (df['close'] - df['low']) / df['total_range'], 0.5)
    
    # 3. 经典K线形态
    df['is_inside'] = (df['high'] <= prev_high) & (df['low'] >= prev_low)
    df['is_outside'] = (df['high'] > prev_high) & (df['low'] < prev_low)
    
    # 4. 大K线（Surprise Bar）
    avg_body = df['body'].rolling(20, min_periods=3).mean()
    df['is_big'] = df['body'] > (avg_body * 1.5)
    
    # 5. ATR
    df['atr'] = df['total_range'].rolling(20, min_periods=3).mean()
    
    return df


def find_swings(df, order=3):
    """纯numpy摆动点识别"""
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
    """
    将图表几何结构与微观博弈完全转化为结构化文本，赋予AI完美的"数字视觉"
    """
    if len(df) < 20: 
        return None, {}
        
    data = df.tail(n_bars).copy().reset_index(drop=True)
    data = compute_bar_features(data)
    
    # 计算当前波动率基准
    current_atr = float(data['atr'].iloc[-1]) if not pd.isna(data['atr'].iloc[-1]) else 10.0
    current_price = float(data['close'].iloc[-1])
    
    # 提取传统的 Swing High / Low 几何高低点
    order = max(2, len(data) // 15)
    sh, sl = find_swings(data, order)
    
    # --- Brooks 专属特征统计 ---
    # 1. 统计最近 10 根 K 线的重叠率
    recent_overlap_ratio = float(data['is_overlap'].tail(10).mean())
    if recent_overlap_ratio >= 0.7:
        overlap_desc = "【极高重叠度】（典型的交易区间 Trading Range 特征，市场无方向）"
    elif recent_overlap_ratio <= 0.3:
        overlap_desc = "【极低重叠度】（典型的强趋势推进特征，动能极强）"
    else:
        overlap_desc = "【中等重叠度】（弱趋势通道或震荡筑底阶段）"
        
    # 2. 统计最近 5 根 K 线的收盘价倾向
    recent_close_pos = float(data['close_pos_ratio'].tail(5).mean())
    if recent_close_pos > 0.65:
        pos_desc = "最近5根K线多收在【高位】（买盘逢低介入坚决）"
    elif recent_close_pos < 0.35:
        pos_desc = "最近5根K线多收在【低位】（卖盘无情打压）"
    else:
        pos_desc = "最近5根K线收盘位置【均衡】（多空博弈剧烈）"
        
    # 3. 统计最近 10 根 K 线的实体性质
    bull_bars = data.tail(10)[data.tail(10)['direction'] == 'bull']
    bear_bars = data.tail(10)[data.tail(10)['direction'] == 'bear']
    big_bars_count = int(data['is_big'].tail(10).sum())
    
    # 寻找最近的几何边界
    dist_to_high = (float(sh[-1][1]) - current_price) if sh else current_atr * 5
    dist_to_low = (current_price - float(sl[-1][1])) if sl else current_atr * 5

    # 组装上下文文本
    context = f"""[当前市场行为扫描]
- 当前价格: {current_price:.2f} | 20周期均幅(ATR): {current_atr:.2f}
- 微观重叠特征: {overlap_desc} (最近10根重叠率: {recent_overlap_ratio:.0%})
- K线微观收盘倾向: {pos_desc}
- 最近10根K线统计: 阳线 {len(bull_bars)} 根，阴线 {len(bear_bars)} 根。其中包含 {big_bars_count} 根异常大K线。
- 距离最近显着局部高点: {dist_to_high:.2f} ({dist_to_high/current_atr:.1f} 倍ATR)
- 距离最近显着局部低点: {dist_to_low:.2f} ({dist_to_low/current_atr:.1f} 倍ATR)

[末尾5根K线明细]
"""
    for i in range(5, 0, -1):
        row = data.iloc[-i]
        t_idx = len(data) - i
        inside_str = " (内包ii)" if row['is_inside'] else ""
        outside_str = " (外包)" if row['is_outside'] else ""
        big_str = " [大K线!]" if row['is_big'] else ""
        context += f"  - 倒数第{i}根K线(K_{t_idx}): {'阳' if row['direction']=='bull' else '阴'}线, 收盘位置比率:{row['close_pos_ratio']:.2%}{inside_str}{outside_str}{big_str}\n"

    struct_data = {
        "swing_highs": sh, "swing_lows": sl, "atr": current_atr, "price": current_price,
        "dist_to_high": dist_to_high, "dist_to_low": dist_to_low
    }
    return context, struct_data


# ==================== AI提示词 ====================

# Step2 批改
STEP2_GRADER = """你是Al Brooks，批改学员的Always In判断。

【图表行为】
{context}

【学员判断】
市场状态：{state}
方向：{direction}
信心：{confidence}
理由：{reason}

【批改】
1. 方向：与当前市场状态（{state}）是否冲突？
2. 信心：{confidence}合理吗？在区间市中是否过于自信？
3. 反方：做反方向最有力的K线是哪根？

【裁决】correct / partial / wrong"""

# Step3 (原 Step4) 批改
STEP3_GRADER = """你是Al Brooks，批改学员的空间判断。

【图表行为】
{context}

【学员判断】
上方空间预估：{upside} ATR
下方风险预估：{downside} ATR
理由：{reason}

【批改】
1. 空间判断是否准确？参考ATR数据
2. 最重要的参照价位是哪根K线？
3. 如果空间不足，正确做法是什么？

【裁决】correct / partial / wrong"""


# Step4 批改
STEP4_GRADER = """你是Al Brooks，批改学员的Signal判断。

【图表行为】
{context}

【学员选择】
信号模式：{signal_mode}
引用K线：{k_ref}

【批改】
1. 该信号是否与当前市场状态和位置匹配？
2. 如果这个信号失败，它通常意味着什么？（维持/减弱/反转）
3. 引用具体K线位置说明理由。

【裁决】correct / partial / wrong"""


STEP5_GRADER = """你是Al Brooks，批改学员的交易计划。

【五步汇总】
市场状态：{state}
Always In：{direction}（{confidence}）
空间判断：{location}
信号：{signal}

【学员计划】
决策：{decision}
胜率类型：{win_rate_type}
放弃条件：{abandon}

【批改】
1. 四个条件都满足吗？
2. 胜率类型与交易类型匹配吗？
3. 放弃条件是否具体？
4. 最终裁决：做/不做/条件性做

【裁决】correct / partial / wrong"""


# 验证终审（AI核心）
VERIFICATION_GRADER = """你是Al Brooks，请根据Price Action原理对学员的交易假设进行终审裁决。

【原始交易假设】
环境状态：{state}
Always In方向：{direction}
设定的放弃条件：{abandon}

【验证期完整K线数据】（共{ver_total}根）
{new_bars}

【学员的自我评判】
他认为交易假设应该：{ver_choice}
理由：{ver_reason}

【Al Brooks裁决标准】
- 横盘（Trading Range）代表趋势的暂停，只要未发生强力的反向突破，原假设就没有被破坏，仅代表动能减弱（对应 partial）。
- 如果学员判断错误（如明明是区间却说趋势维持），请给出正确的判断。

【裁决】correct / partial / wrong
（如学员判断正确则输出correct，反之则输出wrong）"""


COACH_REPORT_PROMPT = """你是Al Brooks首席教练，写训练报告。

【数据】{symbol} {period}，{rounds}轮，{total_min:.0f}分钟
【五步得分】State:{s1}% AI:{s2}% Loc:{s3}% Sig:{s4}% Plan:{s5}%
【假设存活率】{survival:.0f}%
【错误】{errors}

【要求】
1. 最薄弱的1-2步及具体错误
2. 针对性建议
3. 最好的一步
4. 鼓励

300字。"""


# ==================== AI调用 ====================

def get_client():
    key = st.secrets.get("OPENAI_API_KEY", "")
    url = st.secrets.get("OPENAI_BASE_URL", "https://api.deepseek.com")
    model = st.secrets.get("OPENAI_MODEL", "deepseek-chat")
    return OpenAI(base_url=url, api_key=key), model, bool(key)


def call_ai(prompt, max_tokens=500, temp=0.3):
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
        if df is None or len(df) < 100: 
            return None
        # 统一列名
        df = df.rename(columns={
            "date": "time", "open": "open", "high": "high",
            "low": "low", "close": "close", "volume": "volume"
        })
        return df.reset_index(drop=True)
    except Exception as e:
        return None


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

    # 观察截止线
    obs_plot = obs_end - start - 1
    if 0 <= obs_plot < n:
        fig.add_vline(x=obs_plot, line_dash="dash",
                      line_color="#ff9800", line_width=1.5, opacity=0.8)
        fig.add_annotation(x=obs_plot, y=data['high'].max(),
            text="截止", font=dict(size=9, color="#ff9800"), showarrow=False, yshift=14)

    # 验证区高亮（仅在展示时）
    if extra_bars > 0 and obs_plot < n:
        fig.add_vrect(x0=max(0, obs_plot), x1=n-1,
                      fillcolor="#1976d2", opacity=0.06, layer="below", line_width=0)

    # K线编号
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
        "s2_state": "", "s2_evidence": "",
        "s1_dir": "", "s1_confidence": "", "s1_reason": "",
        "s3_upside": "", "s3_downside": "", "s3_reason": "",
        "s4_signal_mode": "", "s4_kref": "",
        "s5_decision": "", "s5_winrate_type": "", "s5_abandon": "",
        # AI批改缓存
        "grade_s2": None, "grade_s1": None, "grade_s3": None,
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
    for k in ["s2_state","s2_evidence",
              "s1_dir","s1_confidence","s1_reason",
              "s3_upside","s3_downside","s3_reason",
              "s4_signal_mode","s4_kref",
              "s5_decision","s5_winrate_type","s5_abandon",
              "grade_s2","grade_s1","grade_s3",
              "grade_s4","grade_s5","grade_ver"]:
        st.session_state[k] = None if k.startswith("grade") else ""
    st.session_state.round_start = time.time()


def load_sym(code, period):
    with st.spinner(f"加载 {code} {PERIODS[period]}..."):
        df = load_data(code, period)
    if df is None or len(df) < 100: 
        return False
    st.session_state.df = df
    st.session_state.symbol = code
    st.session_state.period = period
    return True


def start_round(df, n_bars=60):
    ver_total = random.randint(6, 15)
    max_obs = len(df) - ver_total - 10
    min_obs = n_bars
    if max_obs < min_obs: 
        return False
    obs_end = random.randint(min_obs, max_obs)
    st.session_state.obs_end = obs_end
    st.session_state.ver_total = ver_total
    result = build_behavior_context(df.iloc[:obs_end], n_bars)
    if result[0] is None: 
        return False
    st.session_state.context = result[0]
    st.session_state.struct_data = result[1]
    st.session_state.round_num += 1
    reset_round()
    return True


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
    st.set_page_config(page_title="Al Brooks 心智决策训练器 V5.0", layout="wide")
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
        st.caption("心智决策 · 先判环境")
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
                                st.session_state.phase = "step2"  # 第一步是判环境
                                st.session_state.practice_count += 1
                                st.session_state.session_start = time.time()
                        st.rerun()

        st.markdown("---")
        if st.button("🎲 随机品种", use_container_width=True):
            code = random.choice([c for codes in EXCHANGES.values() for c in codes])
            if load_sym(code, period_sel):
                if start_round(st.session_state.df):
                    st.session_state.phase = "step2"
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
        </div>
