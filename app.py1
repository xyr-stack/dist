import streamlit as st
import numpy as np
import plotly.graph_objects as go
from scipy import stats
import pandas as pd

st.set_page_config(page_title="连续型概率分布交互教学", layout="wide")

st.title("📊 连续型概率分布 · 交互式可视化教学")
st.markdown("---")

# 侧边栏：分布选择与参数控制
with st.sidebar:
    st.header("⚙️ 参数控制")

    dist_type = st.selectbox(
        "选择分布类型",
        ["正态分布", "标准正态分布", "T分布", "卡方分布", "F分布"]
    )

    # 不同分布显示不同参数
    if dist_type == "正态分布":
        mu = st.slider("均值 μ", -5.0, 5.0, 0.0, 0.1)
        sigma = st.slider("标准差 σ", 0.1, 5.0, 1.0, 0.1)
        df_param = None
    elif dist_type == "标准正态分布":
        mu, sigma = 0.0, 1.0
        st.info("标准正态分布：μ=0, σ=1（固定）")
        df_param = None
    elif dist_type == "T分布":
        df_param = st.slider("自由度 df", 1, 50, 10)
        mu, sigma = 0.0, 1.0
    elif dist_type == "卡方分布":
        df_param = st.slider("自由度 df", 1, 50, 5)
        mu, sigma = 0.0, 1.0
    else:  # F分布
        df1 = st.slider("自由度 df1", 1, 50, 5)
        df2 = st.slider("自由度 df2", 1, 50, 20)
        mu, sigma = 0.0, 1.0

    st.markdown("---")
    st.header("📐 概率区间")

    # 区间设置
    col1, col2 = st.columns(2)
    with col1:
        a = st.number_input("区间下限 a", value=-3.0, step=0.5)
    with col2:
        b = st.number_input("区间上限 b", value=3.0, step=0.5)

    show_area = st.checkbox("显示区间概率面积", value=True)

# 主区域：绘图
st.header(f"📈 {dist_type} 概率密度曲线")

# 生成 x 轴范围
if dist_type in ["正态分布", "标准正态分布", "T分布"]:
    x = np.linspace(-8, 8, 1000)
elif dist_type == "卡方分布":
    x = np.linspace(0, max(df_param * 3, 20), 1000)
else:
    x = np.linspace(0, 8, 1000)

# 计算概率密度
if dist_type == "正态分布":
    y = stats.norm.pdf(x, mu, sigma)
    dist_label = f"N({mu}, {sigma}²)"
elif dist_type == "标准正态分布":
    y = stats.norm.pdf(x, 0, 1)
    dist_label = "N(0, 1)"
elif dist_type == "T分布":
    y = stats.t.pdf(x, df_param)
    dist_label = f"t({df_param})"
elif dist_type == "卡方分布":
    y = stats.chi2.pdf(x, df_param)
    dist_label = f"χ²({df_param})"
else:
    y = stats.f.pdf(x, df1, df2)
    dist_label = f"F({df1}, {df2})"

# 创建图表
fig = go.Figure()

# 密度曲线
fig.add_trace(go.Scatter(
    x=x, y=y,
    mode='lines',
    name=dist_label,
    line=dict(color='#1f77b4', width=3)
))

# 概率面积填充
if show_area:
    mask = (x >= a) & (x <= b)
    if np.any(mask):
        fig.add_trace(go.Scatter(
            x=x[mask], y=y[mask],
            mode='lines',
            name=f'P({a} ≤ X ≤ {b})',
            fill='tozeroy',
            line=dict(color='rgba(255,99,71,0)'),
            fillcolor='rgba(255,99,71,0.4)'
        ))

# 计算区间概率
if dist_type == "正态分布":
    prob = stats.norm.cdf(b, mu, sigma) - stats.norm.cdf(a, mu, sigma)
elif dist_type == "标准正态分布":
    prob = stats.norm.cdf(b) - stats.norm.cdf(a)
elif dist_type == "T分布":
    prob = stats.t.cdf(b, df_param) - stats.t.cdf(a, df_param)
elif dist_type == "卡方分布":
    prob = stats.chi2.cdf(b, df_param) - stats.chi2.cdf(a, df_param)
else:
    prob = stats.f.cdf(b, df1, df2) - stats.f.cdf(a, df1, df2)

# 标注均值线（正态分布）
if dist_type in ["正态分布", "标准正态分布"]:
    fig.add_vline(x=mu, line_dash="dash", line_color="green",
                  annotation_text=f"μ={mu}", annotation_position="top")

fig.update_layout(
    xaxis_title="x",
    yaxis_title="概率密度 f(x)",
    height=500,
    showlegend=True,
    hovermode='x'
)

st.plotly_chart(fig, use_container_width=True)

# 统计信息展示
st.markdown("---")
st.header("📊 统计特征")

col1, col2, col3, col4 = st.columns(4)

if dist_type == "正态分布":
    mean_val, var_val = mu, sigma**2
    skew_val, kurt_val = 0, 0
elif dist_type == "标准正态分布":
    mean_val, var_val = 0, 1
    skew_val, kurt_val = 0, 0
elif dist_type == "T分布":
    mean_val = 0 if df_param > 1 else "不存在"
    var_val = df_param/(df_param-2) if df_param > 2 else "不存在"
    skew_val, kurt_val = 0, "6/(df-4)" if df_param > 4 else "不存在"
elif dist_type == "卡方分布":
    mean_val, var_val = df_param, 2*df_param
    skew_val = np.sqrt(8/df_param)
    kurt_val = 12/df_param
else:
    mean_val = df2/(df2-2) if df2 > 2 else "不存在"
    var_val = 2*df2**2*(df1+df2-2)/(df1*(df2-2)**2*(df2-4)) if df2 > 4 else "不存在"
    skew_val = "正偏"
    kurt_val = "—"

col1.metric("均值", mean_val)
col2.metric("方差", var_val)
col3.metric("偏度", skew_val)
col4.metric("峰度", kurt_val)

st.markdown(f"### 🎯 区间概率 P({a} ≤ X ≤ {b}) = **{prob:.4f}**")

# 教学要点
st.markdown("---")
st.header("📚 教学要点")

if dist_type == "正态分布":
    st.markdown("""
    - **形态特征**：钟形曲线，左右对称，由 μ 决定中心位置，σ 决定胖瘦
    - **68-95-99.7法则**：μ±σ 覆盖约68%，μ±2σ 覆盖约95%，μ±3σ 覆盖约99.7%
    - **统计推断用途**：描述自然现象、测量误差、样本均值的分布（中心极限定理）
    """)
elif dist_type == "标准正态分布":
    st.markdown("""
    - **定义**：μ=0、σ=1 的正态分布，记为 N(0,1)
    - **标准化**：任意正态分布 X~N(μ,σ²) 可通过 Z=(X-μ)/σ 转化为标准正态
    - **统计推断用途**：查表计算概率、构建置信区间、假设检验的基础
    """)
elif dist_type == "T分布":
    st.markdown("""
    - **形态特征**：对称钟形，尾部比正态分布更厚（肥尾）
    - **自由度影响**：df 越小尾部越厚，df→∞ 时趋近标准正态分布
    - **统计推断用途**：小样本（n<30）且总体方差未知时的均值检验、置信区间
    """)
elif dist_type == "卡方分布":
    st.markdown("""
    - **形态特征**：右偏分布，取值非负，df 越大越接近正态
    - **定义
    """)
