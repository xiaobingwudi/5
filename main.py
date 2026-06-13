"""
Al Brooks 实盘逐K心智决策训练器 V5.0 (修正稳定版)
核心：先判环境（State），再定方向（Always In）
验证：由AI终审裁决，完美对齐 Price Action 逻辑
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

FIVE_STEPS = ["Market State", "Always In", "Location", "Signal", "Trade Plan"]
CONFIDENCE_LEVELS = ["51%（勉强偏向）", "55%", "60%", "70%", "80%以上（强烈确信）"]
WIN_RATE_TYPES = ["高胜率 / 低盈亏比（如区间反转）", "中等胜率 / 中等盈亏比（如趋势回调）", "低胜率 / 高盈亏比（如突破交易）"]
VER_OPTIONS = ["✅ 交易假设维持（逻辑仍然成立）", "⚠️ 交易假设减弱（仍持有但需观察）", "❌ 交易假设失效（需要退出或反向）"]
SIGNAL_OPTIONS = ["无明显信号（放弃）", "趋势K线（Trend Bar）", "反转棒（Reversal Bar）", "内包棒（Inside Bar）",
                  "H1 / H2（多头一买/二买）", "L1 / L2（空头一卖/二卖）", "ii（连续内包）", "IOI（内-外-内）"]

# ==================== 1. 结构预处理与特征提取 ====================

def compute_bar_features(df):
    df = df.copy()
    df.columns = [c.lower() for c in df.columns]
    
    df['body'] = abs(df['close'] - df['open'])
    df['total_range'] = df['high'] - df['low']
    df['body_ratio'] = np.where(df['total_range'] > 0, df['body'] / df['total_range'], 0)
    df['direction'] = np.where(df['close'] >= df['open'], 'bull', 'bear')
    
    df['upper_shadow'] = df['high'] - df[['open', 'close']].max(axis=1)
    df['lower_shadow'] = df[['open', 'close']].min(axis=1) - df['low']
    
    prev_high = df['high'].shift(1)
    prev_low = df['low'].shift(1)
    overlap_height = np.minimum(df['high'], prev_high) - np.maximum(df['low'], prev_low)
    df['is_overlap'] = overlap_height > 0
    
    df['close_pos_ratio'] = np.where(df['total_range'] > 0, (df['close'] - df['low']) / df['total_range'], 0.5)
    df['is_inside'] = (df['high'] <= prev_high) & (df['low'] >= prev_low)
    df['is_outside'] = (df['high'] > prev_high) & (df['low'] < prev_low)
    
    avg_body = df['body'].rolling(20, min_periods=3).mean()
    df['is_big'] = df['body'] > (avg_body * 1.5)
    df['atr'] = df['total_range'].rolling(20, min_periods=3).mean()
    
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
                if cmp_fn(p[1], result[-1][1]): result[-1] = p
            else: result.append(p)
        return result
    return dedup(sh, lambda a, b: a > b), dedup(sl, lambda a, b: a < b)

def build_behavior_context(df, n_bars=60):
    if len(df) < 20: return None, {}
    data = df.tail(n_bars).copy().reset_index(drop=True)
    data = compute_bar_features(data)
    
    current_atr = float(data['atr'].iloc[-1]) if not pd.isna(data['atr'].iloc[-1]) else 10.0
    current_price = float(data['close'].iloc[-1])
    
    order = max(2, len(data) // 15)
    sh, sl = find_swings(data, order)
    
    recent_overlap_ratio = float(data['is_overlap'].tail(10).mean())
    if recent_overlap_ratio >= 0.7:
        overlap_desc = "【极高重叠度】（典型的交易区间 Trading Range 特征，市场无明显方向）"
    elif recent_overlap_ratio <= 0.3:
        overlap_desc = "【极低重叠度】（典型的强趋势推进特征，动能非对称性极强）"
    else:
        overlap_desc = "【中等重叠度】（弱趋势通道或震荡筑底阶段）"
        
    recent_close_pos = float(data['close_pos_ratio'].tail(5).mean())
    if recent_close_pos > 0.65:
        pos_desc = "最近5根K线多收在【高位】（买盘逢低介入坚决）"
    elif recent_close_pos < 0.35:
        pos_desc = "最近5根K线多收在【低位】（卖盘无情打压）"
    else:
        pos_desc = "最近5根K线收盘位置【均衡】（多空博弈剧烈）"
        
    bull_bars = data.tail(10)[data.tail(10)['direction'] == 'bull']
    bear_bars = data.tail(10)[data.tail(10)['direction'] == 'bear']
    big_bars_count = int(data['is_big'].tail(10).sum())
    
    dist_to_high = (float(sh[-1][1]) - current_price) if sh else current_atr * 5
    dist_to_low = (current_price - float(sl[-1][1])) if sl else current_atr * 5

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

    return context, {"swing_highs": sh, "swing_lows": sl, "atr": current_atr, "price": current_price}

# ==================== AI 提示词 ====================
STEP2_GRADER = """你是Al Brooks。请严格根据Price Action理论批改学员对环境与方向的判断。
【图表行为】\n{context}
【学员判断】\n市场状态：{state}\nAlways In方向：{direction}\n信心等级：{confidence}\n理由：{reason}
【批改要求】\n1. 方向：在此环境状态（{state}）下选这个方向合理吗？\n2. 信心：在区间市中高信心通常是错的，请评判其信心是否合理。\n3. 指出做反方向最有力的那一根K线证据。\n【裁决】correct / partial / wrong"""

STEP3_GRADER = """你是Al Brooks。请批改学员对空间与风险的预估。
【图表行为】\n{context}
【学员判断】\n上方空间预估：{upside} ATR | 下方风险预估：{downside} ATR\n依据理由：{reason}
【批改】\n1. 空间预估是否准确？结合局部高低點和ATR进行数学评判。\n2. 应该以哪一个显着的高低点或K线作为风控基准？\n【裁决】correct / partial / wrong"""

STEP4_GRADER = """你是Al Brooks。请批改学员选择的信号K线。
【图表行为】\n{context}
【学员选择】\n信号模式：{signal_mode} | 引用K线：{k_ref}
【批改】\n1. 該信號模式是否在當前環境下具備高勝率？\n2. 如果該信號隨後失敗了，在Brooks體系中往往預示著什麼機會？\n【裁决】correct / partial / wrong"""

STEP5_GRADER = """你是Al Brooks。请批改交易计划的自洽性。
【五步汇总】\n环境方向：{state} | {direction}\n空间信号：{location} | {signal}
【学员计划】\n决策：{decision} | 胜率类型：{win_rate_type}\n放弃/离场条件：{abandon}
【批改】\n1. 计划与前面的环境、信号是否完全自洽？\n2. 设定的放弃条件是否足够具体到特定K线或价位突破？\n【裁决】correct / partial / wrong"""

VERIFICATION_GRADER = """你是Al Brooks，請根據Price Action原理對學員的交易假設進行【終審裁決】。
【原始交易假設】\n环境状态：{state} | Always In方向：{direction} | 设定的放弃条件：{abandon}
【验证期实际走势】（共{ver_total}根新K线）\n{new_bars}
【学员的自我评判】\n他认为后市走势证明他的假设应该：{ver_choice}\n理由：{ver_reason}

【Al Brooks 终审标准】
- 横盘（Trading Range）代表趋势的暂停，只要未发生强力的反向突破，原假设就没有被破坏，仅代表动能减弱（对应 partial）。
- 仔细核对学判定。若学员自我评判与实际行情演化相符，输出correct；若盲目乐观/悲观，输出wrong。
【裁决】correct / partial / wrong"""

COACH_REPORT_PROMPT = """你是Al Brooks首席教练。根据学员完成的{rounds}轮实战逐K测试，生成300字以内的专业心智复盘报告。
得分率：环境与方向:{s1}% | 空间预估:{s2}% | 信号卡定:{s3}% | 计划自洽:{s4}%
假设终审正确率: {survival:.0f}%
【要求】指出他最严重的1個心智盲點，給出1個具體的Brooks訓練動作，並予以交易員式的硬核鼓勵。"""

# ==================== AI 调用 ====================
def get_client():
    key = st.secrets.get("OPENAI_API_KEY", "")
    url = st.secrets.get("OPENAI_BASE_URL", "https://api.deepseek.com")
    model = st.secrets.get("OPENAI_MODEL", "deepseek-chat")
    return OpenAI(base_url=url, api_key=key), model, bool(key)

def call_ai(prompt, max_tokens=500, temp=0.3):
    client, model, has_key = get_client()
    if not has_key: return "（未配置 API_KEY，请在 .streamlit/secrets.toml 中配置）"
    try:
        r = client.chat.completions.create(model=model, messages=[{"role": "user", "content": prompt}], temperature=temp, max_tokens=max_tokens)
        return r.choices[0].message.content
    except Exception as e: return f"AI调用失败：{str(e)[:100]}"

def parse_grade_and_score(ai_response, step_name):
    for line in reversed(ai_response.strip().split("\n")):
        line = line.strip()
        if "【裁决】" in line or "[裁决]" in line or "裁决" in line:
            if "correct" in line.lower(): st.session_state.scores[step_name].append(1.0)
            elif "partial" in line.lower(): st.session_state.scores[step_name].append(0.5)
            else: st.session_state.scores[step_name].append(0.0)
            return
    st.session_state.scores[step_name].append(0.0)

# ==================== 数据与图表 ====================
def load_data(symbol, period):
    try:
        df = ak.futures_zh_minute_sina(symbol=f"{symbol}0", period=period)
        if df is None or len(df) < 100: return None
        df.columns = [c.lower() for c in df.columns]
        df = df.rename(columns={"date": "time"})
        return df.reset_index(drop=True)
    except: return None

def build_chart(df, obs_end, extra_bars=0, sh=None, sl=None):
    view_end = min(obs_end + extra_bars, len(df))
    start = max(0, view_end - 80)
    data = df.iloc[start:view_end].copy().reset_index(drop=True)
    offset = start

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_heights=[0.75, 0.25])
    fig.add_trace(go.Candlestick(x=data.index, open=data['open'], high=data['high'], low=data['low'], close=data['close'], showlegend=False, increasing_line_color="#ef5350", decreasing_line_color="#26a69a"), row=1, col=1)
    
    ema = data['close'].ewm(span=20).mean()
    fig.add_trace(go.Scatter(x=data.index, y=ema, line=dict(color="#ff9800", width=1.2, dash="dot"), name="EMA20"), row=1, col=1)

    if sh:
        for idx, val in sh:
            if 0 <= idx - offset < len(data):
                fig.add_annotation(x=idx-offset, y=val, text="H", font=dict(size=9, color="#ef5350"), showarrow=False, yshift=10, row=1, col=1)
    if sl:
        for idx, val in sl:
            if 0 <= idx - offset < len(data):
                fig.add_annotation(x=idx-offset, y=val, text="L", font=dict(size=9, color="#26a69a"), showarrow=False, yshift=-12, row=1, col=1)

    obs_plot = obs_end - start - 1
    if 0 <= obs_plot < len(data):
        fig.add_vline(x=obs_plot, line_dash="dash", line_color="#ff9800", line_width=1.5)
    if extra_bars > 0 and obs_plot < len(data):
        fig.add_vrect(x0=max(0, obs_plot), x1=len(data)-1, fillcolor="#1976d2", opacity=0.08, layer="below", line_width=0)

    fig.update_layout(xaxis_rangeslider_visible=False, height=450, margin=dict(l=10, r=10, t=10, b=10), paper_bgcolor="#ffffff", plot_bgcolor="#fafafa")
    return fig

def get_new_bars_text(df, obs_end, n):
    lines = []
    for i in range(n):
        idx = obs_end + i
        if idx >= len(df): break
        row = df.iloc[idx]
        o, h, l, c = row['open'], row['high'], row['low'], row['close']
        d = "阳" if c >= o else "阴"
        lines.append(f"K_{idx+1}: {d}线 [O:{o:.1f} H:{h:.1f} L:{l:.1f} C:{c:.1f}]")
    return "\n".join(lines)

# ==================== Session State ====================
def init_state():
    d = {
        "df": None, "symbol": None, "period": "15", "phase": "select",
        "obs_end": 0, "ver_total": 0, "context": "", "struct_data": {},
        "s2_state": "交易区间 (Trading Range)", "s2_evidence": "",
        "s1_dir": "Always In 多头 (买方占优)", "s1_confidence": "55%", "s1_reason": "",
        "s3_upside": 2.0, "s3_downside": 1.0, "s3_reason": "",
        "s4_signal_mode": ["趋势K线（Trend Bar）"], "s4_kref": "",
        "s5_decision": "顺势做多", "s5_winrate_type": "中等胜率 / 中等盈亏比（如趋势回调）", "s5_abandon": "",
        "ver_choice": "✅ 交易假设维持（逻辑仍然成立）", "ver_reason": "",
        "grade_s2": None, "grade_s1": None, "grade_s3": None, "grade_s4": None, "grade_s5": None, "grade_ver": None,
        "scores": {s: [] for s in FIVE_STEPS}, "verification_scores": [], "round_errors": [], "round_num": 0, "coach_report": None
    }
    for k, v in d.items():
        if k not in st.session_state: st.session_state[k] = v

def start_round(df):
    ver_total = random.randint(6, 15)
    if len(df) - ver_total - 15 < 60: return False
    obs_end = random.randint(60, len(df) - ver_total - 5)
    st.session_state.obs_end = obs_end
    st.session_state.ver_total = ver_total
    ctx, sd = build_behavior_context(df.iloc[:obs_end], 60)
    if ctx is None: return False
    st.session_state.context = ctx
    st.session_state.struct_data = sd
    st.session_state.round_num += 1
    for k in ["grade_s2","grade_s1","grade_s3","grade_s4","grade_s5","grade_ver","s2_evidence","s1_reason","s3_reason","s4_kref","s5_abandon","ver_reason"]:
        st.session_state[k] = None if k.startswith("grade") else ""
    return True

# ==================== CSS 注入 ====================
CSS = """<style>
.stApp { background:#ffffff; color:#222222; }
.step-header { display:flex; align-items:center; gap:12px; background:#f0f4ff; border-left:4px solid #1a56db; padding:10px; margin-bottom:15px; border-radius:4px; }
.step-num { background:#1a56db; color:#fff; font-size:14px; font-weight:700; border-radius:50%; width:26px; height:26px; display:flex; align-items:center; justify-content:center; }
.step-title { font-size:16px; font-weight:700; color:#1a3a8f; }
.ai-box { background:#f0f4ff; border:1px solid #93c5fd; border-radius:6px; padding:12px; margin:10px 0; }
.context-box { background:#fafafa; border:1px solid #ddd; border-radius:4px; padding:10px; font-family:monospace; font-size:12px; white-space:pre-wrap; }
</style>"""

# ==================== 主程序 ====================
def main():
    st.set_page_config(page_title="Al Brooks 心智决策训练器 V5.0", layout="wide")
    st.markdown(CSS, unsafe_allow_html=True)
    init_state()

    # 侧边栏：选择品种与展现得分
    with st.sidebar:
        st.title("Brooks PA 训练沙盘")
        st.caption("心智模型：先判环境，再定方向")
        st.write("---")
        
        if st.session_state.round_num > 0:
            st.write(f"**当前测试进度: 第 {st.session_state.round_num} 轮**")
            for step in FIVE_STEPS:
                scs = st.session_state.scores.get(step, [])
                acc = int(np.mean(scs) * 100) if scs else 0
                st.write(f"· {step}: `{acc}%` 正确率")
            if st.session_state.verification_scores:
                st.write(f"· 终审自洽率: `{int(np.mean(st.session_state.verification_scores)*100)}%`")
        
        st.write("---")
        period_sel = st.selectbox("选择图表周期", list(PERIODS.keys()), format_func=lambda x: PERIODS[x], index=1)
        for cat, codes in EXCHANGES.items():
            with st.expander(cat):
                for code in codes:
                    if st.button(f"📊 {SYMBOL_NAMES[code]} ({code})", key=f"btn_{code}", use_container_width=True):
                        if load_sym(code, period_sel):
                            if start_round(st.session_state.df): st.session_state.phase = "step2"
                        st.rerun()

    # 主操作区路由
    phase = st.session_state.phase
    if phase == "select":
        st.info("💡 请在左侧选择任意商品期货/股指期货品种，系统将随机截取一段历史K线开启盲盒训练。")
        return

    # 渲染图表 (Step 1-5 阶段严格隐藏后市，extra_bars=0)
    is_verifying = (phase in ["verify", "result"])
    eb = st.session_state.ver_total if is_verifying else 0
    fig = build_chart(st.session_state.df, st.session_state.obs_end, extra_bars=eb, sh=st.session_state.struct_data.get("swing_highs"), sl=st.session_state.struct_data.get("swing_lows"))
    st.plotly_chart(fig, use_container_width=True)

    # 展现量化上下文
    with st.expander("🔍 展开当前K线微观多空博弈数据 (数字视觉)", expanded=True):
        st.markdown(f'<div class="context-box">{st.session_state.context}</div>', unsafe_allow_html=True)

    # 阶段分流交互
    if phase == "step2":
        st.markdown('<div class="step-header"><div class="step-num">1</div><div class="step-title">Market State (判断当前市场处于什么容器)</div></div>', unsafe_allow_html=True)
        st.session_state.s2_state = st.radio("当前市场状态", ["强趋势推进 (Spike / Strong Trend)", "弱趋势通道 (Channel / Weak Trend)", "交易区间 (Trading Range / 横盘震荡)"])
        st.session_state.s2_evidence = st.text_area("请给出你的PA证据（如重叠度、均线偏离、高低点抬升情况）:")
        if st.button("提交环境判定，进入方向选择"):
            st.session_state.phase = "step1"
            st.rerun()

    elif phase == "step1":
        st.markdown('<div class="step-header"><div class="step-num">2</div><div class="step-title">Always In (如果被迫必须持仓，你站哪边？)</div></div>', unsafe_allow_html=True)
        st.write(f"先前判定的环境：`{st.session_state.s2_state}`")
        st.session_state.s1_dir = st.radio("Always In 方向", ["Always In 多头 (买方占优)", "Always In 空头 (卖方占优)", "没有任何倾向 (完全无方向区间)"])
        st.session_state.s1_confidence = st.select_slider("你的确信度等级", options=CONFIDENCE_LEVELS)
        st.session_state.s1_reason = st.text_area("请用1句话阐述你站此方向的核心痛点或反方力竭证据:")
        if st.button("提交方向，进入空间评估"):
            # 顺便触发 AI 批改
            p = STEP2_GRADER.format(context=st.session_state.context, state=st.session_state.s2_state, direction=st.session_state.s1_dir, confidence=st.session_state.s1_confidence, reason=st.session_state.s1_reason)
            st.session_state.grade_s2 = call_ai(p)
            parse_grade_and_score(st.session_state.grade_s2, "Always In")
            st.session_state.phase = "step3"
            st.rerun()

    elif phase == "step3":
        st.markdown('<div class="step-header"><div class="step-num">3</div><div class="step-title">Location (空间与风控盈亏比预估)</div></div>', unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        st.session_state.s3_upside = col1.number_input("预估至上方局部阻力处的安全获利空间 (ATR倍数)", min_value=0.0, max_value=10.0, value=2.0, step=0.5)
        st.session_state.s3_downside = col2.number_input("预估跌破至反向保护位需承受的风险 (ATR倍数)", min_value=0.1, max_value=10.0, value=1.0, step=0.5)
        st.session_state.s3_reason = st.text_area("你选择的高点阻力或低点支撑的K线参照依据是？")
        if st.button("提交空间评估，寻找信号棒"):
            p = STEP3_GRADER.format(context=st.session_state.context, upside=st.session_state.s3_upside, downside=st.session_state.s3_downside, reason=st.session_state.s3_reason)
            st.session_state.grade_s3 = call_ai(p)
            parse_grade_and_score(st.session_state.grade_s3, "Location")
            st.session_state.phase = "step4"
            st.rerun()

    elif phase == "step4":
        st.markdown('<div class="step-header"><div class="step-num">4</div><div class="step-title">Signal (卡定当前的微观信号K线形态)</div></div>', unsafe_allow_html=True)
        st.session_state.s4_signal_mode = st.multiselect("当前临近K线触发了哪些 Brooks 经典信号棒模式？", SIGNAL_OPTIONS, default=["趋势K线（Trend Bar）"])
        st.session_state.s4_kref = st.text_area("请指定具体是哪根K线（如倒数第1根、或某局部高低點組合成的H2），并写出其引线或收盘特征：")
        if st.button("提交信号，制定最终交易计划"):
            p = STEP4_GRADER.format(context=st.session_state.context, signal_mode=",".join(st.session_state.s4_signal_mode), k_ref=st.session_state.s4_kref)
            st.session_state.grade_s4 = call_ai(p)
            parse_grade_and_score(st.session_state.grade_s4, "Signal")
            st.session_state.phase = "step5"
            st.rerun()

    elif phase == "step5":
        st.markdown('<div class="step-header"><div class="step-num">5</div><div class="step-title">Trade Plan (交易计划与执行)</div></div>', unsafe_allow_html=True)
        st.session_state.s5_decision = st.radio("你的最终操作决策", ["顺势做多", "逆势高抛低吸", "完全放弃，继续空仓观望"])
        st.session_state.s5_winrate_type = st.selectbox("你此笔交易所依托的交易员方程类型", WIN_RATE_TYPES)
        st.session_state.s5_abandon = st.text_area("【极为重要】你的放弃/离场条件是什么？（一旦发生什么，证明你的假设彻底破灭？）")
        if st.button("封存交易计划，揭晓后市盲盒！"):
            loc_summary = f"盈亏比空间: {st.session_state.s3_upside}对{st.session_state.s3_downside}"
            sig_summary = ",".join(st.session_state.s4_signal_mode)
            p = STEP5_GRADER.format(state=st.session_state.s2_state, direction=st.session_state.s1_dir, confidence=st.session_state.s1_confidence, location=loc_summary, signal=sig_summary, decision=st.session_state.s5_decision, win_rate_type=st.session_state.s5_winrate_type, abandon=st.session_state.s5_abandon)
            st.session_state.grade_s5 = call_ai(p)
            parse_grade_and_score(st.session_state.grade_s5, "Trade Plan")
            st.session_state.phase = "verify"
            st.rerun()

    elif phase == "verify":
        st.markdown("### 🎲 后市盲盒已揭晓！请对比图表蓝色高亮区进行【自我评判】")
        st.warning("请观察右侧新长出的 K 線走勢。在你當初設定的放棄條件觸發前，行情是如期運行，還是陷入了橫盤或被反殺？")
        st.session_state.ver_choice = st.radio("你认为你的原始交易假设目前处于什么状态？", VER_OPTIONS)
        st.session_state.ver_reason = st.text_area("请结合后市长出的具体K线（如发生了惊喜突围、或连续重叠）阐述你的复盘理由：")
        
        if st.button("提交复盘，查看 Al Brooks 终审裁决与教练点评"):
            new_bars_txt = get_new_bars_text(st.session_state.df, st.session_state.obs_end, st.session_state.ver_total)
            p = VERIFICATION_GRADER.format(state=st.session_state.s2_state, direction=st.session_state.s1_dir, abandon=st.session_state.s5_abandon, ver_total=st.session_state.ver_total, new_bars=new_bars_txt, ver_choice=st.session_state.ver_choice, ver_reason=st.session_state.ver_reason)
            st.session_state.grade_ver = call_ai(p)
            
            # 判断终审是否正确
            if "correct" in st.session_state.grade_ver.lower(): st.session_state.verification_scores.append(1.0)
            else: st.session_state.verification_scores.append(0.0)
            st.session_state.phase = "result"
            st.rerun()

    elif phase == "result":
        st.success("🏁 本轮盲盒心智推演已全部结束，以下是 Al Brooks 灵魂导师给你的全套逐步批改报告：")
        
        # 逐步展现 AI 报告
        with st.expander("1. 查看【环境与方向 (Market State & Always In)】导师批改", expanded=True): st.markdown(f'<div class="ai-box">{st.session_state.grade_s2}</div>', unsafe_allow_html=True)
        with st.expander("2. 查看【空间与风控 (Location)】导师批改", expanded=True): st.markdown(f'<div class="ai-box">{st.session_state.grade_s3}</div>', unsafe_allow_html=True)
        with st.expander("3. 查看【微观信号棒 (Signal)】导师批改", expanded=True): st.markdown(f'<div class="ai-box">{st.session_state.grade_s4}</div>', unsafe_allow_html=True)
        with st.expander("4. 查看【计划自洽性 (Trade Plan)】导师批改", expanded=True): st.markdown(f'<div class="ai-box">{st.session_state.grade_s5}</div>', unsafe_allow_html=True)
        with st.expander("5. 👑 【后市验证·终审裁决】", expanded=True): st.markdown(f'<div class="ai-box" style="border: 2px solid #22c55e;">{st.session_state.grade_ver}</div>', unsafe_allow_html=True)

        if st.button("开启下一轮随机盲盒测试", type="primary"):
            if start_round(st.session_state.df): st.session_state.phase = "step2"
            st.rerun()

if __name__ == "__main__":
    main()
