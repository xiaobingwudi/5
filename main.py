"""
Al Brooks 日内机会寻找训练器 V5.0
核心架构：故事更新器（Story Evolution）

设计理念：
- 不是事后解剖80根K线，而是逐根重演当天
- 每根新K线出现时，用户需要更新对市场故事的理解
- AI不断追问：什么证据支持？什么证据反对？什么情况会推翻？
- 最终输出完整的故事时间轴

核心模块：
1. 事件提取器 - 检测每根K线发生的事实（不解释）
2. 故事更新器 - 根据新事件更新当前故事（核心）
3. 证据挑战器 - 每次更新后自动追问证据和反证
4. 故事时间轴 - 记录整个演化过程
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

# ==================== 配置参数 ====================
STORY_EVOLUTION_STEPS = [5, 15, 30, 45, 60, 80]  # 故事更新的K线节点

# ==================== 事件提取器 ====================
def extract_events(df, start_idx, end_idx):
    """
    提取指定K线范围内发生的关键事件
    只记录事实，不解释
    """
    sub = df.iloc[start_idx:end_idx+1]
    if len(sub) < 2:
        return []
    
    events = []
    
    # 1. 连续同向K线（≥3根）
    streak = 1
    streak_dir = None
    for i in range(len(sub)-1, 0, -1):
        curr, prev = sub.iloc[i], sub.iloc[i-1]
        curr_bull = curr["close"] >= curr["open"]
        prev_bull = prev["close"] >= prev["open"]
        if streak_dir is None:
            streak_dir = curr_bull
            streak = 1
        elif curr_bull == streak_dir:
            streak += 1
        else:
            break
    if streak >= 3:
        direction = "多头" if streak_dir else "空头"
        events.append(f"连续{streak}根{direction}K线（无明显重叠）")
    
    # 2. 大K线（实体>80%波幅）
    last = sub.iloc[-1]
    body = abs(last["close"] - last["open"])
    total = last["high"] - last["low"]
    body_ratio = body / total if total > 0 else 0
    if body_ratio > 0.80:
        direction = "阳线" if last["close"] >= last["open"] else "阴线"
        events.append(f"大{direction}线（实体占比{body_ratio*100:.0f}%）")
    
    # 3. 内包/外包线
    if len(sub) >= 2:
        prev, curr = sub.iloc[-2], sub.iloc[-1]
        if curr["high"] < prev["high"] and curr["low"] > prev["low"]:
            events.append(f"内包线(IB)")
        elif curr["high"] > prev["high"] and curr["low"] < prev["low"]:
            events.append(f"外包线(OB)")
    
    # 4. 创区间新高/新低
    recent_lows = [row["low"] for _, row in sub.iterrows()]
    recent_highs = [row["high"] for _, row in sub.iterrows()]
    current_low = last["low"]
    current_high = last["high"]
    
    if current_high >= max(recent_highs) - 0.01:
        events.append(f"创区间新高")
    if current_low <= min(recent_lows) + 0.01:
        events.append(f"创区间新低")
    
    return events


# ==================== 故事更新器（AI核心）====================
STORY_UPDATE_SYSTEM = """你是 Al Brooks 价格行为教练。

【你的任务】
用户正在逐根K线复盘当天的走势。每一段结束后，用户会描述当前的市场故事。
你需要：
1. 总结用户描述的故事
2. 追问：什么证据支持这个故事？
3. 追问：什么证据反对这个故事？
4. 追问：如果这个故事错了，最可能是什么情况？

【输出格式】
- 先说"好的，我理解当前故事是：[用户的描述]"
- 然后问2-3个引导性问题
- 控制在150字以内

【当前段信息】
K线范围: K{start} ~ K{end}
事件: {events}
"""


def call_story_updater(segment_start, segment_end, events, user_story, conversation_history):
    """调用故事更新器"""
    api_key = st.secrets.get("OPENAI_API_KEY", "")
    base_url = st.secrets.get("OPENAI_BASE_URL", "https://api.deepseek.com")
    model = st.secrets.get("OPENAI_MODEL", "deepseek-chat")
    
    if not api_key:
        return "【提示】请配置API密钥\n\n请描述你的观察..."
    
    system = STORY_UPDATE_SYSTEM.format(
        start=segment_start,
        end=segment_end,
        events="、".join(events) if events else "无明显特殊事件"
    )
    
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": f"用户描述的故事：{user_story}"}
    ]
    
    try:
        client = OpenAI(base_url=base_url, api_key=api_key)
        resp = client.chat.completions.create(
            model=model, messages=messages, temperature=0.3, max_tokens=300
        )
        return resp.choices[0].message.content
    except Exception as e:
        return f"AI错误: {e}\n\n继续训练..."


# ==================== 故事时间轴 ====================
def format_timeline(timeline):
    """格式化输出故事时间轴"""
    if not timeline:
        return "暂无记录"
    lines = []
    for entry in timeline:
        lines.append(f"**【K{entry['end']}】** {entry['story']}")
    return "\n\n".join(lines)


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
def build_chart(df, current_bar, highlight_bars=None):
    """绘制K线图，可高亮指定区域"""
    end = current_bar + 1
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

    # 标记K线编号
    for idx, orig_idx in enumerate(original_indices):
        if orig_idx % 10 == 0:
            row_data = plot_df.iloc[idx]
            fig.add_annotation(
                x=idx, y=row_data["low"],
                text=str(orig_idx), showarrow=False,
                font=dict(size=7, color="#888888"),
                yshift=-14, row=1, col=1
            )
    
    # 高亮当前段
    if highlight_bars:
        for h_start, h_end in highlight_bars:
            if h_start >= start and h_end <= end:
                fig.add_vrect(
                    x0=h_start - start, x1=h_end - start,
                    fillcolor="lightblue", opacity=0.15,
                    layer="below", line_width=0
                )
    
    # 标记当前K线
    bar_pos = original_indices.index(current_bar) if current_bar in original_indices else len(original_indices) - 1
    fig.add_vline(x=bar_pos, line_dash="dash", line_color="#ff9800", line_width=1.5, opacity=0.8)

    fig.update_layout(
        xaxis_rangeslider_visible=False,
        height=500,
        margin=dict(l=5, r=5, t=5, b=5),
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
        "current_bar": 80,
        "timeline": [],  # 故事时间轴
        "current_stage": 0,  # 当前阶段索引
        "conversations": {},  # 每个阶段的对话历史
        "segment_stories": {},  # 每个阶段用户记录的故事
        "training_complete": False,
        "practice_count": 0,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def reset_training():
    st.session_state.timeline = []
    st.session_state.current_stage = 0
    st.session_state.conversations = {}
    st.session_state.segment_stories = {}
    st.session_state.training_complete = False


def load_new_symbol(code, period_value):
    with st.spinner(f"加载 {code} {period_value}分钟..."):
        df = load_data(f"{code}0", period=period_value)
        if df is None or len(df) < 100:
            st.error(f"{code} 数据加载失败")
            return False
        
        st.session_state.df = df
        st.session_state.symbol = code
        st.session_state.current_bar = STORY_EVOLUTION_STEPS[0]  # 从第一段开始
        reset_training()
        return True


# ==================== 主界面 ====================
def main():
    st.set_page_config(page_title="Al Brooks 故事更新训练器 V5.0", layout="wide")
    
    # 浅色系CSS
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
    .stSuccess {
        background-color: #d1e7dd;
        color: #0f5132;
    }
    </style>
    """, unsafe_allow_html=True)

    init_state()

    # 侧边栏
    with st.sidebar:
        st.markdown("### 📊 故事更新训练器 V5.0")
        st.markdown("---")
        st.markdown("**训练理念**")
        st.caption("不是事后解剖80根K线，而是逐根重演当天。")
        st.caption("每根新K线出现时，更新对市场故事的理解。")
        st.caption("AI会追问：什么证据支持？什么证据反对？")
        
        st.markdown("---")
        st.markdown("**完成次数**")
        st.metric("完整复盘次数", st.session_state.practice_count)
        
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
            if st.button("🔄 重置当前训练", use_container_width=True):
                reset_training()
                st.rerun()
            
            total_bars = len(st.session_state.df)
            st.caption(f"数据范围: {total_bars}根K线 | 当前阶段: {st.session_state.current_stage+1}/{len(STORY_EVOLUTION_STEPS)}")

    # 主界面
    if st.session_state.df is None:
        st.markdown("## 👈 请从左侧选择品种开始训练")
        st.markdown("""
        <div style="background:#f8f9fa;padding:20px;border-radius:12px;border:1px solid #e9ecef;">
        <h3>Al Brooks 故事更新训练器 V5.0</h3>
        <p><strong>这不是形态识别器，而是故事更新器。</strong></p>
        <p>训练流程：</p>
        <ol>
            <li>系统会按阶段推进K线（K5 → K15 → K30 → K45 → K60 → K80）</li>
            <li>每个阶段结束后，你需要描述当前的市场故事</li>
            <li>AI会追问：什么证据支持？什么证据反对？什么情况会推翻？</li>
            <li>全部6个阶段完成后，输出完整的故事时间轴</li>
        </ol>
        <p style="color:#6c757d;">💡 训练的是“快速更新对市场故事的理解”，而不是“识别形态”。</p>
        </div>
        """, unsafe_allow_html=True)
        return

    df = st.session_state.df
    current_stage = st.session_state.current_stage
    total_stages = len(STORY_EVOLUTION_STEPS)
    
    # 训练完成
    if st.session_state.training_complete:
        st.success("🎉 恭喜！完成一次完整复盘！")
        
        st.markdown("### 📖 故事时间轴")
        st.markdown(format_timeline(st.session_state.timeline))
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🔄 开始下一次复盘", type="primary"):
                st.session_state.practice_count += 1
                reset_training()
                st.rerun()
        with col2:
            if st.button("🎲 换一个品种"):
                # 随机选择一个新品种
                all_codes = []
                for codes in EXCHANGES.values():
                    all_codes.extend(codes)
                new_code = random.choice(all_codes)
                load_new_symbol(new_code, "30")
                st.rerun()
        return
    
    # 当前阶段
    if current_stage >= total_stages:
        st.session_state.training_complete = True
        st.rerun()
        return
    
    current_end_bar = STORY_EVOLUTION_STEPS[current_stage]
    prev_end_bar = STORY_EVOLUTION_STEPS[current_stage - 1] if current_stage > 0 else 0
    
    # 计算高亮区域
    highlight_bars = []
    for i, end in enumerate(STORY_EVOLUTION_STEPS):
        if i <= current_stage:
            start_prev = STORY_EVOLUTION_STEPS[i-1] if i > 0 else 0
            highlight_bars.append((start_prev, end))
    
    # 显示图表
    st.plotly_chart(build_chart(df, current_end_bar, highlight_bars), use_container_width=True)
    
    # 显示进度
    st.progress((current_stage) / total_stages, text=f"阶段 {current_stage+1}/{total_stages}：K{prev_end_bar} → K{current_end_bar}")
    
    # 显示当前段的事件
    events = extract_events(df, prev_end_bar, current_end_bar)
    if events:
        with st.expander("📌 本段关键事件（系统检测，仅作参考）", expanded=False):
            for e in events:
                st.markdown(f"- {e}")
    
    # 显示已有的故事时间轴
    if st.session_state.timeline:
        with st.expander("📖 已有故事时间轴", expanded=False):
            st.markdown(format_timeline(st.session_state.timeline))
    
    # 对话区域
    conv_key = f"stage_{current_stage}"
    if conv_key not in st.session_state.conversations:
        st.session_state.conversations[conv_key] = []
    
    conv = st.session_state.conversations[conv_key]
    
    # 显示历史对话
    for msg in conv:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
    
    # 如果当前阶段还没有故事记录
    if conv_key not in st.session_state.segment_stories:
        # 等待用户输入故事
        st.markdown(f"### 🎯 K{prev_end_bar} → K{current_end_bar}")
        st.markdown("**请描述当前的市场故事：**")
        st.markdown("- 谁在控制市场？")
        st.markdown("- 故事是否发生了变化？")
        st.markdown("- 什么证据支持你的判断？")
        
        user_input = st.chat_input("输入你对当前故事的描述...")
        
        if user_input:
            conv.append({"role": "user", "content": user_input})
            st.session_state.segment_stories[conv_key] = user_input
            
            # 调用AI故事更新器
            with st.spinner("AI分析中..."):
                ai_response = call_story_updater(
                    prev_end_bar, current_end_bar, events, user_input, conv
                )
            
            conv.append({"role": "assistant", "content": ai_response})
            st.rerun()
    
    else:
        # 已有故事记录，显示确认按钮
        st.info(f"📝 当前故事：{st.session_state.segment_stories[conv_key]}")
        
        if st.button("✅ 确认，进入下一阶段", type="primary"):
            # 记录到时间轴
            story_summary = st.session_state.segment_stories[conv_key][:200]
            st.session_state.timeline.append({
                "end": current_end_bar,
                "story": story_summary
            })
            
            # 进入下一阶段
            st.session_state.current_stage += 1
            st.rerun()


if __name__ == "__main__":
    main()
