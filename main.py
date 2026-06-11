"""
Al Brooks 日内机会寻找训练器 V12.2
修复：K线编号统一从1开始，与图表显示一致
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

PERIODS = ["5", "15", "30", "60"]
PERIOD_LABELS = {"5": "5分钟", "15": "15分钟", "30": "30分钟", "60": "60分钟"}

# 每页显示的K线数量
DISPLAY_BARS = 60

# ==================== 数据加载 ====================
@st.cache_data(ttl=1800, show_spinner=False)
def load_futures_data(symbol, period):
    try:
        df = ak.futures_zh_minute_sina(symbol=f"{symbol}0", period=period)
        if df is None or len(df) < 30:
            return None
        df = df.rename(columns={
            "date": "time", "open": "open", "high": "high",
            "low": "low", "close": "close", "volume": "volume"
        })
        return df.reset_index(drop=True)
    except Exception as e:
        return None


# ==================== 图表绘制（编号从1开始）====================
def build_chart(df, start_display_idx=0, current_display_pos=None, highlight_end=None):
    """
    绘制K线图
    - start_display_idx: 从原始数据中的哪个索引开始显示（0表示第一根）
    - current_display_pos: 当前观察到的位置（显示编号，从1开始）
    - highlight_end: 高亮到哪个显示编号
    """
    # 取要显示的数据段
    end_idx = min(start_display_idx + DISPLAY_BARS, len(df))
    plot_df = df.iloc[start_display_idx:end_idx].copy().reset_index(drop=True)
    n_bars = len(plot_df)
    
    # 显示编号：从1开始，不管原始索引是多少
    display_numbers = list(range(1, n_bars + 1))

    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        vertical_spacing=0.02, row_heights=[0.75, 0.25]
    )

    # K线
    fig.add_trace(go.Candlestick(
        x=plot_df.index,
        open=plot_df['open'], high=plot_df['high'],
        low=plot_df['low'], close=plot_df['close'],
        showlegend=False,
        increasing_line_color="#ef5350",
        decreasing_line_color="#26a69a",
    ), row=1, col=1)

    # 每根K线都标注编号
    for idx, bar_num in enumerate(display_numbers):
        row = plot_df.iloc[idx]
        # 阳线：编号标在下方；阴线：编号标在上方
        if row['close'] >= row['open']:
            y_pos = row['low']
            y_shift = -12
        else:
            y_pos = row['high']
            y_shift = 12
        
        fig.add_annotation(
            x=idx, y=y_pos,
            text=f"K{bar_num}",
            showarrow=False,
            font=dict(size=8, color="#666666"),
            yshift=y_shift,
            row=1, col=1
        )
    
    # 每10根用更明显的标记突出显示
    for idx, bar_num in enumerate(display_numbers):
        if bar_num % 10 == 0:
            row = plot_df.iloc[idx]
            if row['close'] >= row['open']:
                y_pos = row['low']
                y_shift = -22
            else:
                y_pos = row['high']
                y_shift = 22
            fig.add_annotation(
                x=idx, y=y_pos,
                text=f"★ K{bar_num}",
                showarrow=False,
                font=dict(size=10, color="#ff9800", weight="bold"),
                yshift=y_shift,
                row=1, col=1
            )

    # 标记当前位置（显示编号对应的图表位置）
    if current_display_pos is not None and 1 <= current_display_pos <= n_bars:
        fig.add_vline(
            x=current_display_pos - 1, line_dash="dash",
            line_color="#ff9800", line_width=2, opacity=0.8
        )
        fig.add_annotation(
            x=current_display_pos - 1, y=plot_df.iloc[current_display_pos - 1]['high'],
            text="← 当前位置", showarrow=False,
            font=dict(size=10, color="#ff9800"),
            yshift=15
        )
    
    # 高亮观察范围（从K1到highlight_end）
    if highlight_end is not None and highlight_end <= n_bars:
        fig.add_vrect(
            x0=0, x1=highlight_end - 1,
            fillcolor="#4caf50", opacity=0.1,
            layer="below", line_width=0
        )

    # 成交量
    vol_colors = ["#ef5350" if c >= o else "#26a69a"
                  for o, c in zip(plot_df['open'], plot_df['close'])]
    fig.add_trace(go.Bar(
        x=plot_df.index, y=plot_df['volume'],
        marker_color=vol_colors, showlegend=False, opacity=0.5
    ), row=2, col=1)

    fig.update_layout(
        xaxis_rangeslider_visible=False,
        height=520,
        margin=dict(l=10, r=10, t=30, b=10),
        paper_bgcolor="#ffffff",
        plot_bgcolor="#f8f9fa",
        font=dict(color="#333333"),
    )
    fig.update_xaxes(showgrid=True, gridcolor="#e0e0e0", gridwidth=0.5, showticklabels=False)
    fig.update_yaxes(showgrid=True, gridcolor="#e0e0e0", gridwidth=0.5)

    return fig


# ==================== AI 挑战提示词 ====================
CHALLENGE_PROMPT = """你是 Al Brooks 价格行为教练。

【你的角色】
你不是老师，你是陪练。你的任务是挑战用户的观察，而不是判断对错。

【用户观察的K线范围】
K1 ~ K{end}

【用户认为市场在做什么】
{observation}

【用户的判断依据】
{evidence}

【用户认为什么信号会证明他错了】
{fail_signal}

【你的任务】
请从以下角度选1-2个提问：
1. 引用具体K线编号，指出用户可能遗漏的关键K线
2. 追问反方视角：如果做相反方向，最可能依据哪根K线？
3. 追问验证条件：如果接下来出现什么行为，你会改变判断？

【输出要求】
- 不要判断对错
- 不要给出结论
- 控制在100字以内
"""

VERIFICATION_PROMPT = """你是 Al Brooks 价格行为教练。

【用户之前的判断】
{observation}

【用户认为会证明他错的信号】
{fail_signal}

【新出现的K线行为】
{new_bars}

【你的任务】
请帮助用户思考：新出现的K线是否改变了他的判断？

控制在60字以内。
"""


def call_challenge(end_bar, observation, evidence, fail_signal):
    """AI挑战用户"""
    api_key = st.secrets.get("OPENAI_API_KEY", "")
    base_url = st.secrets.get("OPENAI_BASE_URL", "https://api.deepseek.com")
    model = st.secrets.get("OPENAI_MODEL", "deepseek-chat")
    
    if not api_key:
        return "请引用具体K线补充你的观察。"
    
    prompt = CHALLENGE_PROMPT.format(
        end=end_bar,
        observation=observation[:200],
        evidence=evidence[:200],
        fail_signal=fail_signal[:200]
    )
    
    try:
        client = OpenAI(base_url=base_url, api_key=api_key)
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.5,
            max_tokens=200
        )
        return resp.choices[0].message.content
    except Exception as e:
        return f"AI连接失败: {str(e)[:50]}"


def call_verification(observation, fail_signal, new_bars_text):
    """验证新K线"""
    api_key = st.secrets.get("OPENAI_API_KEY", "")
    base_url = st.secrets.get("OPENAI_BASE_URL", "https://api.deepseek.com")
    model = st.secrets.get("OPENAI_MODEL", "deepseek-chat")
    
    if not api_key:
        return "新出现的K线是否改变了你的判断？"
    
    prompt = VERIFICATION_PROMPT.format(
        observation=observation[:200],
        fail_signal=fail_signal[:200],
        new_bars=new_bars_text[:200]
    )
    
    try:
        client = OpenAI(base_url=base_url, api_key=api_key)
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.5,
            max_tokens=150
        )
        return resp.choices[0].message.content
    except Exception as e:
        return f"新出现的K线是否改变了你的判断？(API: {str(e)[:50]})"


# ==================== Session State ====================
def init_state():
    defaults = {
        "df": None,
        "symbol": None,
        "period": "15",
        "phase": "select",           # select / observe / challenge / verify / complete
        "start_display_idx": 0,      # 从原始数据的哪个索引开始显示
        "current_display_pos": 30,   # 当前观察到的位置（显示编号，从1开始）
        "user_observation": "",
        "user_evidence": "",
        "user_fail_signal": "",
        "ai_challenge": "",
        "verification_result": "",
        "practice_count": 0,
        "current_practice_start": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def reset_training():
    st.session_state.phase = "observe"
    st.session_state.user_observation = ""
    st.session_state.user_evidence = ""
    st.session_state.user_fail_signal = ""
    st.session_state.ai_challenge = ""
    st.session_state.verification_result = ""


def load_symbol(code, period):
    with st.spinner(f"加载 {code} {PERIOD_LABELS[period]}数据..."):
        df = load_futures_data(code, period)
    if df is None or len(df) < 50:
        return False
    
    st.session_state.df = df
    st.session_state.symbol = code
    st.session_state.period = period
    # 从最新的数据中取一段，确保有足够K线
    st.session_state.start_display_idx = max(0, len(df) - DISPLAY_BARS)
    st.session_state.current_display_pos = 30
    reset_training()
    return True


# ==================== 主界面 ====================
def main():
    st.set_page_config(page_title="Al Brooks 读盘训练器", layout="wide")
    
    st.markdown("""
    <style>
    .stApp { background-color: #ffffff; }
    [data-testid="stSidebar"] {
        background-color: #f5f5f5 !important;
        border-right: 1px solid #e0e0e0;
    }
    h1, h2, h3, p, .stMarkdown {
        color: #333333 !important;
    }
    .stButton > button {
        background: #f0f0f0;
        border: 1px solid #cccccc;
        color: #333333;
        border-radius: 6px;
    }
    .stButton > button:hover {
        background: #e0e0e0;
        border-color: #999999;
    }
    .stTextArea textarea {
        background-color: #fafafa;
        border-color: #dddddd;
    }
    .stProgress > div > div {
        background-color: #4caf50;
    }
    .info-box {
        background: #f8f9fa;
        border: 1px solid #e0e0e0;
        border-radius: 8px;
        padding: 16px;
        margin: 12px 0;
    }
    .info-box-title {
        color: #4caf50;
        font-size: 13px;
        font-weight: 600;
        margin-bottom: 8px;
    }
    </style>
    """, unsafe_allow_html=True)

    init_state()

    # 侧边栏
    with st.sidebar:
        st.markdown("## 📊 Al Brooks 读盘训练器")
        st.markdown("---")
        
        st.markdown("**选择品种**")
        period = st.selectbox(
            "K线周期",
            options=PERIODS,
            format_func=lambda x: PERIOD_LABELS[x],
            index=1
        )
        
        for cat, codes in EXCHANGES.items():
            with st.expander(cat, expanded=False):
                cols = st.columns(2)
                for idx, code in enumerate(codes):
                    name = SYMBOL_NAMES.get(code, code)
                    if cols[idx % 2].button(f"{code}", key=f"btn_{code}", use_container_width=True):
                        if load_symbol(code, period):
                            st.rerun()
                        else:
                            st.error(f"{code} 数据加载失败")
        
        st.markdown("---")
        if st.button("🎲 随机品种", use_container_width=True):
            all_codes = [c for codes in EXCHANGES.values() for c in codes]
            code = random.choice(all_codes)
            if load_symbol(code, period):
                st.rerun()
        
        st.markdown("---")
        st.markdown(f"**训练次数**")
        st.markdown(f"<div style='font-size:28px;font-weight:600;color:#4caf50;'>{st.session_state.practice_count}</div>", unsafe_allow_html=True)

    # 选择品种页面
    if st.session_state.df is None:
        st.markdown("## 👈 从左侧选择品种开始训练")
        st.markdown("""
        <div class="info-box">
            <div class="info-box-title">📖 训练流程</div>
            <div style="color:#333333;line-height:1.8;">
                1. 观察K1到K30的K线图<br>
                2. 回答三个问题：
                   - 市场在做什么？<br>
                   - 你的判断依据是什么？（引用具体K线编号）<br>
                   - 如果判断错误，会出现什么信号？<br>
                3. AI会挑战你的观察（不是判断对错）<br>
                4. 推进10根新K线，验证你的判断<br>
                5. 完成一次完整的读盘训练
            </div>
        </div>
        """, unsafe_allow_html=True)
        return

    df = st.session_state.df
    period_label = PERIOD_LABELS[st.session_state.period]
    symbol = st.session_state.symbol

    # 训练完成
    if st.session_state.phase == "complete":
        st.success("🎉 完成一次读盘训练！")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🔄 继续训练", type="primary"):
                st.session_state.practice_count += 1
                # 随机换一段数据
                max_start = max(0, len(df) - DISPLAY_BARS)
                new_start = random.randint(0, max_start)
                st.session_state.start_display_idx = new_start
                st.session_state.current_display_pos = 30
                reset_training()
                st.rerun()
        with col2:
            if st.button("🎲 换一个品种"):
                all_codes = [c for codes in EXCHANGES.values() for c in codes]
                code = random.choice([c for c in all_codes if c != symbol])
                if load_symbol(code, st.session_state.period):
                    st.rerun()
        return

    # 观察阶段
    if st.session_state.phase == "observe":
        # 显示图表（K1到K60）
        st.plotly_chart(
            build_chart(
                df,
                start_display_idx=st.session_state.start_display_idx,
                current_display_pos=st.session_state.current_display_pos,
                highlight_end=st.session_state.current_display_pos
            ),
            use_container_width=True
        )
        
        st.markdown(f"### 🔍 观察 K1 → K{st.session_state.current_display_pos}")
        st.info("💡 提示：图表中每根K线都有编号（K1、K2、K3...），请引用具体编号说明你的判断依据。")
        
        with st.form(key="observation_form"):
            observation = st.text_area(
                "① 市场在做什么？",
                placeholder="例如：空头在控制，连续出现阴线，反弹很弱...",
                height=80
            )
            
            evidence = st.text_area(
                "② 你的判断依据是什么？（请引用具体的K线编号）",
                placeholder="例如：K15-K20连续5根阴线；K22反弹失败；K25收盘新低...",
                height=80
            )
            
            fail_signal = st.text_area(
                "③ 如果判断错误，会出现什么信号？",
                placeholder="例如：如果出现连续两根阳线收盘在K22高点上方，我会重新评估...",
                height=80
            )
            
            submitted = st.form_submit_button("✅ 提交观察", type="primary")
            
            if submitted:
                if observation and evidence:
                    st.session_state.user_observation = observation
                    st.session_state.user_evidence = evidence
                    st.session_state.user_fail_signal = fail_signal
                    
                    with st.spinner("AI思考中..."):
                        challenge = call_challenge(
                            st.session_state.current_display_pos,
                            observation, evidence, fail_signal
                        )
                    st.session_state.ai_challenge = challenge
                    st.session_state.phase = "challenge"
                    st.rerun()
                else:
                    st.warning("请至少填写前两个问题")

    # AI挑战阶段
    elif st.session_state.phase == "challenge":
        st.plotly_chart(
            build_chart(
                df,
                start_display_idx=st.session_state.start_display_idx,
                current_display_pos=st.session_state.current_display_pos,
                highlight_end=st.session_state.current_display_pos
            ),
            use_container_width=True
        )
        
        st.markdown(f"### 🤖 AI教练挑战")
        
        with st.expander("📋 你的观察", expanded=True):
            st.markdown(f"**市场行为：** {st.session_state.user_observation}")
            st.markdown(f"**判断依据：** {st.session_state.user_evidence}")
            if st.session_state.user_fail_signal:
                st.markdown(f"**反证信号：** {st.session_state.user_fail_signal}")
        
        with st.chat_message("assistant"):
            st.markdown(st.session_state.ai_challenge)
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("✏️ 修改观察", use_container_width=True):
                st.session_state.phase = "observe"
                st.rerun()
        with col2:
            if st.button("➡️ 继续，进入验证", type="primary", use_container_width=True):
                st.session_state.phase = "verify"
                st.rerun()

    # 验证阶段（推进10根K线）
    elif st.session_state.phase == "verify":
        new_pos = min(st.session_state.current_display_pos + 10, DISPLAY_BARS)
        
        st.plotly_chart(
            build_chart(
                df,
                start_display_idx=st.session_state.start_display_idx,
                current_display_pos=new_pos,
                highlight_end=new_pos
            ),
            use_container_width=True
        )
        
        st.markdown(f"### 🔍 验证阶段：K{st.session_state.current_display_pos+1} → K{new_pos}")
        st.markdown("新出现的K线是否改变了你的判断？")
        
        # 显示新K线的简要描述
        # 需要从数据中提取对应的K线
        new_bars_text = ""
        start_idx = st.session_state.start_display_idx
        for i in range(st.session_state.current_display_pos, new_pos):
            data_idx = start_idx + i
            if data_idx < len(df):
                row = df.iloc[data_idx]
                bar_num = i + 1
                direction = "阳" if row['close'] >= row['open'] else "阴"
                new_bars_text += f"K{bar_num}: {direction}线，开={row['open']:.0f} 高={row['high']:.0f} 低={row['low']:.0f} 收={row['close']:.0f}\n"
        
        with st.expander("📊 新K线详情", expanded=True):
            st.text(new_bars_text if new_bars_text else "无新K线数据")
        
        if st.session_state.user_fail_signal:
            st.info(f"💡 你之前设定的反证信号：{st.session_state.user_fail_signal}")
        
        verification = st.text_area(
            "新出现的K线是否改变了你的判断？请说明原因。",
            placeholder="例如：没有改变，空头仍然控制，因为反弹很弱...",
            height=100
        )
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("↩️ 返回修改", use_container_width=True):
                st.session_state.phase = "observe"
                st.rerun()
        with col2:
            if st.button("✅ 完成验证", type="primary", use_container_width=True):
                if verification:
                    with st.spinner("AI分析中..."):
                        ai_response = call_verification(
                            st.session_state.user_observation,
                            st.session_state.user_fail_signal,
                            new_bars_text
                        )
                    st.session_state.verification_result = verification
                    st.session_state.ai_verification = ai_response
                    st.session_state.current_display_pos = new_pos
                    st.session_state.phase = "complete"
                    st.rerun()
                else:
                    st.warning("请描述你的验证结果")


if __name__ == "__main__":
    main()
