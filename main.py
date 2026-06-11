"""
Al Brooks 日内机会寻找训练器 V11.0
核心目标：观察行为 → 多剧本概率 → 行为预期 → 逐根验证 → 更新概率

核心理念：
- 不是解释市场，而是观察市场做了什么
- 不是给方向加权，而是给剧本概率
- 每根K线都会改变概率
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

# 周期配置（基于国内期货6-8小时交易时间，观察窗口时间接近）
PERIOD_CONFIG = {
    "5分钟": {"period": "5", "display": "5分钟", "obs_window": 12, "ver_window": 3, "max_bars": 80, "desc": "12根=60分钟"},
    "15分钟": {"period": "15", "display": "15分钟", "obs_window": 8, "ver_window": 2, "max_bars": 32, "desc": "8根=120分钟（推荐）"},
    "30分钟": {"period": "30", "display": "30分钟", "obs_window": 4, "ver_window": 1, "max_bars": 16, "desc": "4根=120分钟"},
    "60分钟": {"period": "60", "display": "60分钟", "obs_window": 2, "ver_window": 1, "max_bars": 8, "desc": "2根=120分钟"},
}

# 验证选项（行为清单）
VERIFICATION_CHECKLIST = [
    "出现跟进买方K线",
    "出现跟进卖方K线",
    "突破成功并维持",
    "突破失败",
    "回调浅且被迅速买回",
    "回调深且未修复",
    "测试前高但未突破",
    "测试前低但未跌破",
    "进入窄幅横盘",
    "出现反向吞没K线",
    "出现长影线反转信号",
    "无明确新行为"
]

# ==================== AI 提示词 ====================
CHALLENGE_SYSTEM = """你是 Al Brooks 价格行为教练。

【你的角色】
你不是老师，你是陪练。
你的任务不是判断对错，而是挑战用户的观察和剧本。

【最近10根K线数据】
{recent_bars}

【用户的观察】（市场做了什么）
{behavior}

【用户的剧本和概率】
{scenarios}

【输出格式】
请从以下角度选2个提问：
1. 引用具体K线编号，指出用户可能遗漏的行为
2. 追问反方视角：如果做相反方向，最可能依据哪根K线？
3. 追问放弃条件：如果出现什么行为，你会放弃当前最可能的剧本？

不要泛泛提问。
不要替用户分析。
控制在100字以内。
"""

VERIFICATION_SYSTEM = """你是 Al Brooks 价格行为教练。

【上一段的预期】
{expectation}

【用户选择的验证项】
{checked_items}

【输出格式】
请追问1个问题：
- 这些验证项中，哪个对你更新概率最重要？为什么？

控制在50字以内。
"""

SUMMARY_CHALLENGE_SYSTEM = """你是 Al Brooks 价格行为教练。

【用户的总结】
{user_summary}

【决策记录】
{decision_log}

【输出格式】
请追问1-2个帮助用户发现遗漏的问题：
- 哪一次概率调整最重要？
- 哪根K线最出乎意料？

控制在80字以内。
"""


def call_challenge(behavior, scenarios, recent_bars_text, step_num):
    """AI挑战用户的观察和剧本"""
    api_key = st.secrets.get("OPENAI_API_KEY", "")
    base_url = st.secrets.get("OPENAI_BASE_URL", "https://api.deepseek.com")
    model = st.secrets.get("OPENAI_MODEL", "deepseek-chat")
    
    if not api_key:
        return "请引用具体K线补充你的观察。"
    
    # 格式化剧本
    scenarios_text = ""
    for i, s in enumerate(scenarios):
        scenarios_text += f"剧本{i+1}: {s.get('desc', '')} ({s.get('prob', 0)}%)\n"
    
    system = CHALLENGE_SYSTEM.format(
        recent_bars=recent_bars_text,
        behavior=behavior[:300],
        scenarios=scenarios_text
    )
    
    try:
        client = OpenAI(base_url=base_url, api_key=api_key)
        resp = client.chat.completions.create(
            model=model, messages=[{"role": "system", "content": system}],
            temperature=0.5, max_tokens=150
        )
        return resp.choices[0].message.content
    except Exception as e:
        return f"请补充具体K线的观察。(API: {str(e)[:50]})"


def call_verification(expectation, checked_items):
    """验证反馈"""
    api_key = st.secrets.get("OPENAI_API_KEY", "")
    base_url = st.secrets.get("OPENAI_BASE_URL", "https://api.deepseek.com")
    model = st.secrets.get("OPENAI_MODEL", "deepseek-chat")
    
    if not api_key:
        return "哪个验证项对你更新概率最重要？"
    
    system = VERIFICATION_SYSTEM.format(
        expectation=expectation,
        checked_items="、".join(checked_items) if checked_items else "无"
    )
    
    try:
        client = OpenAI(base_url=base_url, api_key=api_key)
        resp = client.chat.completions.create(
            model=model, messages=[{"role": "system", "content": system}],
            temperature=0.5, max_tokens=80
        )
        return resp.choices[0].message.content
    except Exception as e:
        return f"哪个验证项最重要？(API: {str(e)[:50]})"


def call_summary_challenge(user_summary, decision_log):
    """挑战用户的总结"""
    api_key = st.secrets.get("OPENAI_API_KEY", "")
    base_url = st.secrets.get("OPENAI_BASE_URL", "https://api.deepseek.com")
    model = st.secrets.get("OPENAI_MODEL", "deepseek-chat")
    
    if not api_key:
        return "哪一次概率调整最重要？"
    
    # 格式化决策日志
    log_text = ""
    for log in decision_log[-5:]:
        log_text += f"K{log['end']}: {log.get('scenarios', [])}\n"
    
    system = SUMMARY_CHALLENGE_SYSTEM.format(
        user_summary=user_summary[:500],
        decision_log=log_text[:1000]
    )
    
    try:
        client = OpenAI(base_url=base_url, api_key=api_key)
        resp = client.chat.completions.create(
            model=model, messages=[{"role": "system", "content": system}],
            temperature=0.5, max_tokens=120
        )
        return resp.choices[0].message.content
    except Exception as e:
        return "哪一次概率调整最重要？"


# ==================== 数据加载 ====================
@st.cache_data(ttl=3600, show_spinner=False)
def load_data(symbol, period="15"):
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
def build_chart(df, max_bar, current_pos=None, obs_start=None, obs_end=None, decision_log=None):
    """绘制K线图"""
    start = max(0, len(df) - max_bar)
    plot_df = df.iloc[start:].copy().reset_index(drop=True)
    n_bars = len(plot_df)
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

    # 每5根显示一次编号
    for idx, bar_num in enumerate(bar_numbers):
        if bar_num % 5 == 0:
            row_data = plot_df.iloc[idx]
            if row_data["close"] >= row_data["open"]:
                y_pos = row_data["low"]
                y_shift = -22
            else:
                y_pos = row_data["high"]
                y_shift = 22
            fig.add_annotation(
                x=idx, y=y_pos,
                text=f"K{bar_num}",
                showarrow=False,
                font=dict(size=9, color="#666666"),
                yshift=y_shift,
                row=1, col=1
            )

    # 标记当前位置
    if current_pos and current_pos <= n_bars:
        fig.add_vline(
            x=current_pos - 1, line_dash="dash",
            line_color="#ff9800", line_width=2, opacity=0.8
        )
    
    # 标记观察窗口
    if obs_start and obs_end:
        if obs_start >= start and obs_end <= start + n_bars:
            fig.add_vrect(
                x0=obs_start - start, x1=obs_end - start,
                fillcolor="#4caf50", opacity=0.1,
                layer="below", line_width=0
            )

    # 标记概率变化
    if decision_log:
        for log in decision_log:
            pos = log.get("end")
            if pos and pos <= start + n_bars:
                prob = log.get("top_prob", "")
                fig.add_annotation(
                    x=pos - start - 1, y=plot_df.iloc[min(pos-start-1, n_bars-1)]["low"],
                    text=f"{prob}%", showarrow=False,
                    font=dict(size=8, color="#9c27b0"),
                    yshift=-25
                )

    fig.update_layout(
        xaxis_rangeslider_visible=False,
        height=500,
        margin=dict(l=10, r=10, t=10, b=10),
        paper_bgcolor="#ffffff",
        plot_bgcolor="#f8f9fa",
        font=dict(color="#212529"),
    )
    fig.update_xaxes(showgrid=True, gridcolor="#e9ecef", gridwidth=0.5)
    fig.update_yaxes(showgrid=True, gridcolor="#e9ecef", gridwidth=0.5)
    
    return fig


# ==================== Session State ====================
def init_state():
    defaults = {
        "df": None,
        "symbol": None,
        "period_config": PERIOD_CONFIG["15分钟"],
        "max_bar": 32,
        "current_pos": 0,
        "obs_start": 0,
        "obs_end": 0,
        "ver_count": 0,
        "decision_log": [],
        "session_complete": False,
        "practice_count": 0,
        "phase": "observation",  # observation / challenge / verification / update / summary
        "current_observation": None,
        "current_scenarios": [],
        "current_expectation": "",
        "temp_data": {},
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def reset_session():
    st.session_state.current_pos = 0
    st.session_state.obs_start = 0
    st.session_state.obs_end = 0
    st.session_state.ver_count = 0
    st.session_state.decision_log = []
    st.session_state.session_complete = False
    st.session_state.phase = "observation"
    st.session_state.current_observation = None
    st.session_state.current_scenarios = []
    st.session_state.current_expectation = ""
    st.session_state.temp_data = {}


def load_new_symbol(code, period_label):
    config = PERIOD_CONFIG[period_label]
    with st.spinner(f"加载 {code} {config['period']}分钟..."):
        df = load_data(f"{code}0", period=config["period"])
        if df is None or len(df) < config["max_bars"]:
            st.error(f"{code} 数据不足，需要至少{config['max_bars']}根K线")
            return False
        
        st.session_state.df = df
        st.session_state.symbol = code
        st.session_state.period_config = config
        st.session_state.max_bar = min(config["max_bars"], len(df))
        reset_session()
        return True


def get_recent_bars_text(df, current_pos, lookback=10):
    """获取最近10根K线的文本描述"""
    start = max(0, current_pos - lookback)
    sub = df.iloc[start:current_pos]
    if len(sub) == 0:
        return "暂无数据"
    
    lines = []
    for i, row in sub.iterrows():
        o, h, l, c = row["open"], row["high"], row["low"], row["close"]
        body = abs(c - o)
        total = h - l
        body_ratio = body / total * 100 if total > 0 else 0
        direction = "阳" if c >= o else "阴"
        lines.append(f"K{i}: {direction} O={o:.0f} H={h:.0f} L={l:.0f} C={c:.0f} | 实体占比{body_ratio:.0f}%")
    
    return "\n".join(lines)


# ==================== 主界面 ====================
def main():
    st.set_page_config(page_title="Al Brooks V11.0 | 概率更新训练器", layout="wide")
    
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
    .scenario-card {
        background-color: #e3f2fd;
        border-radius: 8px;
        padding: 12px;
        margin: 8px 0;
    }
    .observation-card {
        background-color: #e8f5e9;
        border-radius: 8px;
        padding: 12px;
        margin: 8px 0;
    }
    </style>
    """, unsafe_allow_html=True)

    init_state()

    # 侧边栏
    with st.sidebar:
        st.markdown("### 📊 Al Brooks V11.0")
        st.markdown("**概率更新训练器**")
        st.markdown("---")
        st.markdown("**核心理念**")
        st.caption("观察行为 → 多剧本概率 → 行为预期 → 逐根验证 → 更新概率")
        st.caption("每根K线都会改变概率")
        
        st.markdown("---")
        st.markdown("**周期选择**")
        
        period_label = st.radio(
            "K线周期",
            options=list(PERIOD_CONFIG.keys()),
            format_func=lambda x: f"{x} - {PERIOD_CONFIG[x]['desc']}",
            index=1
        )
        
        if period_label != st.session_state.period_config["display"]:
            if st.session_state.df is not None:
                load_new_symbol(st.session_state.symbol, period_label)
                st.rerun()
        
        st.markdown("---")
        st.metric("完成复盘次数", st.session_state.practice_count)
        
        if st.session_state.decision_log:
            st.markdown("---")
            st.markdown("**📈 概率演化**")
            for log in st.session_state.decision_log[-5:]:
                prob = log.get("top_prob", "?")
                st.caption(f"K{log['end']}: {prob}%")
        
        st.markdown("---")
        st.markdown("**选择品种**")
        for cat, codes in EXCHANGES.items():
            with st.expander(cat):
                cols = st.columns(2)
                for idx, code in enumerate(codes):
                    name = SYMBOL_NAMES.get(code, code)
                    if cols[idx % 2].button(f"{code}\n{name}", key=f"btn_{code}", use_container_width=True):
                        if load_new_symbol(code, period_label):
                            st.rerun()
        
        st.markdown("---")
        if st.session_state.df is not None:
            if st.button("🔄 重置训练", use_container_width=True):
                reset_session()
                st.rerun()

    # 主界面
    if st.session_state.df is None:
        st.markdown("## 👈 请从左侧选择品种开始训练")
        st.markdown(f"""
        <div style="background:#f8f9fa;padding:20px;border-radius:12px;border:1px solid #e9ecef;">
        <h3>Al Brooks 概率更新训练器 V11.0</h3>
        <p><strong>核心理念：每根K线都会改变概率。</strong></p>
        
        <h4>📖 训练流程</h4>
        <ol>
            <li><strong>观察窗口</strong>：观察5-12根K线（取决于周期）</li>
            <li><strong>描述行为</strong>：禁止贴标签，只描述市场做了什么</li>
            <li><strong>构建剧本</strong>：写3个最合理的剧本，分配概率（总和100%）</li>
            <li><strong>形成预期</strong>：如果最可能的剧本正确，接下来应该看到什么行为</li>
            <li><strong>逐根验证</strong>：每根新K线验证行为清单</li>
            <li><strong>更新概率</strong>：根据验证结果更新剧本概率</li>
            <li><strong>总结</strong>：今天的故事、最大意外、最重要修正</li>
        </ol>
        
        <p style="color:#6c757d;">💡 Al Brooks: "Every bar changes the probability."</p>
        </div>
        """, unsafe_allow_html=True)
        return

    df = st.session_state.df
    config = st.session_state.period_config
    max_bar = st.session_state.max_bar
    phase = st.session_state.phase
    
    # 显示图表
    st.plotly_chart(build_chart(
        df, max_bar, 
        st.session_state.current_pos if st.session_state.current_pos > 0 else None,
        st.session_state.obs_start if st.session_state.obs_start > 0 else None,
        st.session_state.obs_end if st.session_state.obs_end > 0 else None,
        st.session_state.decision_log
    ), use_container_width=True)
    
    # 进度
    if st.session_state.current_pos > 0:
        progress = st.session_state.current_pos / max_bar
        st.progress(progress, text=f"进度：K{st.session_state.current_pos} / K{max_bar}")
    
    # ========== Phase 1: 观察 ==========
    if phase == "observation":
        obs_window = config["obs_window"]
        
        if st.session_state.obs_end == 0:
            st.session_state.obs_end = min(obs_window, max_bar)
        
        st.markdown(f"### 🔍 观察窗口：K{st.session_state.obs_start+1} → K{st.session_state.obs_end}")
        
        with st.form(key="observation_form"):
            st.markdown("**① 市场刚刚做了什么？**")
            st.caption("（禁止写\"多头控制\"、\"空头占优\"等结论，只描述具体行为）")
            
            behavior = st.text_area(
                "行为描述",
                placeholder="例如：\n- K5-K8连续出现3根阴线，收盘在低位\n- K9出现一根大阳线，但收盘未过前高\n- K10-K12窄幅横盘，成交量萎缩",
                height=120
            )
            
            st.markdown("**② 最合理的3个剧本（含概率，总和100%）**")
            
            scenario1_desc = st.text_input("剧本1描述", placeholder="例如：空头突破成功并出现跟进", key="s1_desc")
            scenario1_prob = st.number_input("剧本1概率", min_value=0, max_value=100, value=50, step=5, key="s1_prob")
            
            scenario2_desc = st.text_input("剧本2描述", placeholder="例如：失败突破，回到区间", key="s2_desc")
            scenario2_prob = st.number_input("剧本2概率", min_value=0, max_value=100, value=30, step=5, key="s2_prob")
            
            scenario3_desc = st.text_input("剧本3描述", placeholder="例如：区间继续震荡", key="s3_desc")
            scenario3_prob = st.number_input("剧本3概率", min_value=0, max_value=100, value=20, step=5, key="s3_prob")
            
            st.markdown("**③ 如果剧本1正确，接下来应该看到什么行为？**")
            expectation = st.selectbox(
                "预期行为",
                options=VERIFICATION_CHECKLIST,
                index=len(VERIFICATION_CHECKLIST) - 1
            )
            
            submitted = st.form_submit_button("✅ 提交观察", type="primary")
            
            if submitted:
                if behavior and scenario1_desc:
                    # 验证概率总和
                    total_prob = scenario1_prob + scenario2_prob + scenario3_prob
                    if total_prob != 100:
                        st.warning(f"概率总和应为100%，当前{total_prob}%")
                    else:
                        scenarios = [
                            {"desc": scenario1_desc, "prob": scenario1_prob},
                            {"desc": scenario2_desc, "prob": scenario2_prob},
                            {"desc": scenario3_desc, "prob": scenario3_prob}
                        ]
                        st.session_state.current_observation = behavior
                        st.session_state.current_scenarios = scenarios
                        st.session_state.current_expectation = expectation
                        st.session_state.phase = "challenge"
                        st.rerun()
                else:
                    st.warning("请填写行为描述和至少1个剧本")
    
    # ========== Phase 2: AI挑战 ==========
    elif phase == "challenge":
        st.markdown("### 🤖 AI挑战")
        
        if "challenge_response" not in st.session_state.temp_data:
            recent_bars_text = get_recent_bars_text(df, st.session_state.obs_end, lookback=10)
            with st.spinner("AI思考中..."):
                response = call_challenge(
                    st.session_state.current_observation,
                    st.session_state.current_scenarios,
                    recent_bars_text,
                    1
                )
            st.session_state.temp_data["challenge_response"] = response
        
        with st.chat_message("assistant"):
            st.markdown(st.session_state.temp_data["challenge_response"])
        
        st.markdown("**你的回应（可选）**")
        user_response = st.text_area("", placeholder="可以补充观察，或确认进入下一步...", height=60)
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("✅ 确认，进入验证", type="primary"):
                st.session_state.phase = "verification"
                st.session_state.ver_count = 0
                st.rerun()
        with col2:
            if st.button("↩️ 返回修改"):
                st.session_state.phase = "observation"
                st.session_state.temp_data.pop("challenge_response", None)
                st.rerun()
    
    # ========== Phase 3: 逐根验证 ==========
    elif phase == "verification":
        ver_window = config["ver_window"]
        obs_end = st.session_state.obs_end
        ver_count = st.session_state.ver_count
        
        if ver_count < ver_window:
            next_bar = obs_end + ver_count + 1
            
            if next_bar > max_bar:
                st.session_state.phase = "update"
                st.rerun()
            
            st.markdown(f"### 🔍 验证第 {ver_count+1}/{ver_window} 根新K线")
            st.markdown(f"**K{next_bar}**")
            
            # 显示新K线详情
            new_bar_data = df.iloc[next_bar-1] if next_bar <= len(df) else None
            if new_bar_data is not None:
                o, h, l, c = new_bar_data["open"], new_bar_data["high"], new_bar_data["low"], new_bar_data["close"]
                body = abs(c - o)
                total = h - l
                body_ratio = body / total * 100 if total > 0 else 0
                direction = "阳线" if c >= o else "阴线"
                st.info(f"**K{next_bar}：{direction}，开盘{o:.0f}，最高{h:.0f}，最低{l:.0f}，收盘{c:.0f}，实体占比{body_ratio:.0f}%**")
            
            st.markdown(f"**预期行为：** {st.session_state.current_expectation}")
            
            # 行为清单验证
            st.markdown("**实际发生了什么？（可多选）**")
            checked_items = []
            for item in VERIFICATION_CHECKLIST[:8]:  # 显示前8个，避免太长
                if st.checkbox(item, key=f"v_{ver_count}_{item}"):
                    checked_items.append(item)
            
            col1, col2 = st.columns(2)
            with col1:
                if st.button("✅ 提交验证", type="primary"):
                    if checked_items:
                        st.session_state.temp_data[f"verification_{ver_count}"] = {
                            "bar": next_bar,
                            "checked": checked_items
                        }
                        st.session_state.ver_count += 1
                        st.rerun()
                    else:
                        st.warning("请至少选择一项")
            
            with col2:
                if st.button("⏭️ 跳过，直接更新概率"):
                    st.session_state.phase = "update"
                    st.rerun()
        
        else:
            st.session_state.phase = "update"
            st.rerun()
    
    # ========== Phase 4: 更新概率 ==========
    elif phase == "update":
        st.markdown("### ⚖️ 更新概率")
        
        # 显示验证汇总
        st.markdown("**验证结果汇总：**")
        for i in range(st.session_state.ver_count):
            v_data = st.session_state.temp_data.get(f"verification_{i}", {})
            if v_data:
                st.markdown(f"- K{v_data['bar']}: {', '.join(v_data['checked'])}")
        
        st.markdown("---")
        st.markdown("**根据验证结果，更新剧本概率：**")
        
        scenarios = st.session_state.current_scenarios
        
        new_prob1 = st.number_input(f"{scenarios[0]['desc'][:30]}", value=scenarios[0]['prob'], step=5, key="new_p1")
        new_prob2 = st.number_input(f"{scenarios[1]['desc'][:30]}", value=scenarios[1]['prob'], step=5, key="new_p2")
        new_prob3 = st.number_input(f"{scenarios[2]['desc'][:30]}", value=scenarios[2]['prob'], step=5, key="new_p3")
        
        total_new = new_prob1 + new_prob2 + new_prob3
        if total_new != 100:
            st.warning(f"概率总和应为100%，当前{total_new}%")
        
        update_reason = st.text_area("为什么这样调整？", placeholder="例如：空头没有出现跟进，降低空头概率...", height=60)
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("✅ 确认更新", type="primary"):
                if total_new == 100:
                    # 记录决策
                    st.session_state.decision_log.append({
                        "end": st.session_state.obs_end + st.session_state.ver_count,
                        "obs_range": f"{st.session_state.obs_start+1}-{st.session_state.obs_end}",
                        "behavior": st.session_state.current_observation[:100],
                        "scenarios": [{"desc": s["desc"], "prob": p} for s, p in zip(scenarios, [new_prob1, new_prob2, new_prob3])],
                        "top_prob": new_prob1,
                        "expectation": st.session_state.current_expectation,
                        "update_reason": update_reason
                    })
                    
                    # 推进到下一轮
                    st.session_state.obs_start = st.session_state.obs_end
                    st.session_state.obs_end = min(st.session_state.obs_start + st.session_state.period_config["obs_window"], max_bar)
                    st.session_state.current_pos = st.session_state.obs_end
                    st.session_state.ver_count = 0
                    st.session_state.phase = "observation"
                    st.session_state.temp_data = {}
                    
                    # 检查是否完成
                    if st.session_state.obs_start >= max_bar:
                        st.session_state.session_complete = True
                    
                    st.rerun()
        
        with col2:
            if st.button("↩️ 重新验证"):
                st.session_state.phase = "verification"
                st.rerun()
    
    # ========== Phase 5: 总结 ==========
    elif phase == "summary":
        st.success("🎉 完成一次读盘训练！")
        
        if st.session_state.decision_log:
            st.markdown("### 📈 概率演化")
            
            # 创建概率演化表格
            prob_data = []
            for log in st.session_state.decision_log:
                prob_data.append({
                    "位置": f"K{log['end']}",
                    "概率": log.get("top_prob", 0)
                })
            prob_df = pd.DataFrame(prob_data)
            st.line_chart(prob_df.set_index("位置"))
        
        st.markdown("---")
        st.markdown("### 📝 总结今天的故事")
        
        summary_q1 = st.text_area("① 今天市场试图做什么？", height=60)
        summary_q2 = st.text_area("② 最终成功了吗？", height=60)
        summary_q3 = st.text_area("③ 最大的意外是什么？", height=60)
        summary_q4 = st.text_area("④ 哪一次概率调整最重要？为什么？", height=60)
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("提交总结", type="primary"):
                if summary_q1:
                    user_summary = f"{summary_q1}\n\n{summary_q2}\n\n{summary_q3}\n\n{summary_q4}"
                    with st.spinner("AI分析中..."):
                        challenge = call_summary_challenge(user_summary, st.session_state.decision_log)
                    st.session_state.summary_challenge = challenge
                    st.rerun()
        
        if "summary_challenge" in st.session_state:
            with st.chat_message("assistant"):
                st.markdown(f"**🤖 AI挑战**\n\n{st.session_state.summary_challenge}")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🔄 继续训练", type="secondary"):
                st.session_state.practice_count += 1
                reset_session()
                st.rerun()
        with col2:
            if st.button("🎲 换一个品种"):
                all_codes = []
                for codes in EXCHANGES.values():
                    all_codes.extend(codes)
                new_code = random.choice(all_codes)
                load_new_symbol(new_code, st.session_state.period_config["display"])
                st.rerun()


if __name__ == "__main__":
    main()
