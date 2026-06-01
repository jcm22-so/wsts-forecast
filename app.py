import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from prophet import Prophet

st.set_page_config(page_title="WSTS Semiconductor Forecast", layout="wide")
st.title("📈 Semiconductor Sales — Time Series & Forecast")

# ── 1. Upload file ──────────────────────────────────────────
uploaded = st.file_uploader("Upload WSTS.xlsx", type=["xlsx"])
if uploaded is None:
    st.stop()

df = pd.read_excel(uploaded)
df.columns = df.columns.str.strip().str.title()
df['Year']  = pd.to_numeric(df['Year'],  errors='coerce')
df['Month'] = pd.to_numeric(df['Month'], errors='coerce')
df.dropna(subset=['Year','Month'], inplace=True)
df['Year']  = df['Year'].astype(int)
df['Month'] = df['Month'].astype(int)
df['Date']  = pd.to_datetime(df[['Year','Month']].assign(Day=1))

# ── 2. Sidebar filters ──────────────────────────────────────
st.sidebar.header("Filters")

view_by = st.sidebar.radio("View by", ["Category", "Region"])

if view_by == "Category":
    options = sorted(df['Category'].unique())
    selected = st.sidebar.multiselect("Select categories", options, default=options)
    group_col = "Category"
else:
    options = sorted(df['Region'].unique())
    selected = st.sidebar.multiselect("Select regions", options, default=options)
    group_col = "Region"

forecast_months = st.sidebar.slider("Forecast horizon (months)", 6, 60, 24)

# ── 3. Filter data ──────────────────────────────────────────
df_filtered = df[df[group_col].isin(selected)]

# ── 4. Time series chart ────────────────────────────────────
st.subheader(f"Monthly Sales by {view_by}")
ts = df_filtered.groupby(['Date', group_col])['Value'].sum().reset_index()
fig = go.Figure()
for grp in selected:
    sub = ts[ts[group_col] == grp]
    fig.add_trace(go.Scatter(x=sub['Date'], y=sub['Value'], name=grp))
fig.update_layout(template='plotly_white')
st.plotly_chart(fig, use_container_width=True)

# ── 5. Forecast ─────────────────────────────────────────────
st.subheader(f"Prophet Forecast by {view_by}")

metrics_list = []

for grp in selected:
    series = (df_filtered[df_filtered[group_col] == grp]
              .groupby('Date')['Value'].sum()
              .rename('Value').asfreq('MS'))

    train = series.iloc[:-12]
    test  = series.iloc[-12:]

    df_p = train.reset_index().rename(columns={'Date':'ds','Value':'y'})
    m = Prophet(yearly_seasonality=True, weekly_seasonality=False,
                daily_seasonality=False, interval_width=0.95)
    m.fit(df_p)

    future = m.make_future_dataframe(periods=12 + forecast_months, freq='MS')
    fc = m.predict(future)

    # Metrics
    pred_test = fc[fc['ds'].isin(test.index)]['yhat'].values
    import numpy as np
    mad  = float(np.mean(np.abs(test.values - pred_test)))
    mse  = float(np.mean((test.values - pred_test)**2))
    mape = float(np.mean(np.abs((test.values - pred_test)/test.values))*100)
    metrics_list.append({group_col: grp, 'MAD': round(mad,0),
                         'MSE': round(mse,0), 'MAPE_%': round(mape,2)})

    # Plot
    cutoff = series.index[-1]
    fut = fc[fc['ds'] > cutoff]
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=series.index, y=series.values, name='Historical'))
    fig2.add_trace(go.Scatter(x=fut['ds'], y=fut['yhat'], name='Forecast',
                              line=dict(color='tomato')))
    fig2.add_traces([go.Scatter(
        x=pd.concat([fut['ds'], fut['ds'][::-1]]),
        y=pd.concat([fut['yhat_upper'], fut['yhat_lower'][::-1]]),
        fill='toself', fillcolor='rgba(255,99,71,0.15)',
        line=dict(color='rgba(0,0,0,0)'), name='95% CI')])
    fig2.update_layout(title=f"{grp} — MAPE: {mape:.1f}%", template='plotly_white')
    st.plotly_chart(fig2, use_container_width=True)

# ── 6. Metrics table ────────────────────────────────────────
st.subheader("Evaluation Metrics")
st.dataframe(pd.DataFrame(metrics_list), use_container_width=True)
