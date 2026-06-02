import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from prophet import Prophet

st.set_page_config(page_title="WSTS Semiconductor Forecast", layout="wide")
st.title("📈 Semiconductor Sales — Time Series & Forecast")

# ── 1. Upload file ──────────────────────────────────────────
uploaded = st.file_uploader("Upload WSTS.xlsx", type=["xlsx"])
if uploaded is None:
    st.info("Please upload your WSTS.xlsx file to continue.")
    st.stop()

df = pd.read_excel(uploaded)
df.columns = df.columns.str.strip().str.title()
df['Category'] = df['Category'].str.strip().str.title()
df['Region']   = df['Region'].str.strip().str.title()
df['Value']    = pd.to_numeric(df['Value'], errors='coerce')
df['Year']     = pd.to_numeric(df['Year'], errors='coerce')

# Convert Month from name to number
month_map = {
    'January':1,  'February':2,  'March':3,     'April':4,
    'May':5,       'June':6,      'July':7,      'August':8,
    'September':9, 'October':10,  'November':11, 'December':12
}
df['Month'] = df['Month'].map(month_map)

df.dropna(subset=['Year','Month','Value'], inplace=True)
df['Year']  = df['Year'].astype(int)
df['Month'] = df['Month'].astype(int)
df['Date']  = pd.to_datetime(df[['Year','Month']].assign(Day=1))

# ── 2. Sidebar filters ──────────────────────────────────────
st.sidebar.header("Filters")

view_by = st.sidebar.radio("View by", ["Category", "Region"])

if view_by == "Category":
    group_col   = "Category"
    all_options = sorted(df['Category'].dropna().unique().tolist())
else:
    group_col   = "Region"
    all_options = sorted(df['Region'].dropna().unique().tolist())

select_all = st.sidebar.checkbox("Select all", value=True)

if select_all:
    selected = all_options
else:
    selected = st.sidebar.multiselect(
        f"Select {view_by.lower()}s",
        options=all_options,
        default=all_options[:1]
    )

forecast_months = st.sidebar.slider("Forecast horizon (months)", 6, 60, 24)

if not selected:
    st.warning("Please select at least one option from the sidebar.")
    st.stop()

# ── 3. Filter data ──────────────────────────────────────────
df_filtered = df[df[group_col].isin(selected)].copy()

# ── 4. Time series chart ────────────────────────────────────
st.subheader(f"Monthly Sales by {view_by}")
ts = df_filtered.groupby(['Date', group_col])['Value'].sum().reset_index()

fig = go.Figure()
for grp in selected:
    sub = ts[ts[group_col] == grp]
    fig.add_trace(go.Scatter(x=sub['Date'], y=sub['Value'],
                             mode='lines', name=grp))
fig.update_layout(template='plotly_white',
                  xaxis_title='Date', yaxis_title='Sales',
                  legend=dict(orientation='h'))
st.plotly_chart(fig, use_container_width=True)

# ── 5. Forecast ─────────────────────────────────────────────
st.subheader(f"Prophet Forecast by {view_by}")
st.info(f"Running forecast for {len(selected)} group(s) — this may take a moment...")

metrics_list = []

for grp in selected:
    series = (df_filtered[df_filtered[group_col] == grp]
              .groupby('Date')['Value'].sum()
              .rename('Value')
              .asfreq('MS'))

    if len(series) < 24:
        st.warning(f"'{grp}' has less than 24 months of data — skipping.")
        continue

    train = series.iloc[:-12]
    test  = series.iloc[-12:]

    df_p = train.reset_index().rename(columns={'Date':'ds','Value':'y'})

    m = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=False,
        daily_seasonality=False,
        interval_width=0.95
    )
    m.fit(df_p)

    future = m.make_future_dataframe(periods=12 + forecast_months, freq='MS')
    fc     = m.predict(future)

    # ── Metrics ──
    pred_test = fc[fc['ds'].isin(test.index)]['yhat'].values
    if len(pred_test) == len(test):
        mad  = float(np.mean(np.abs(test.values - pred_test)))
        mse  = float(np.mean((test.values - pred_test) ** 2))
        mape = float(np.mean(np.abs((test.values - pred_test) / test.values)) * 100)
    else:
        mad, mse, mape = None, None, None

    metrics_list.append({
        group_col:  grp,
        'MAD':      round(mad,  0) if mad  is not None else 'N/A',
        'MSE':      round(mse,  0) if mse  is not None else 'N/A',
        'MAPE_%':   round(mape, 2) if mape is not None else 'N/A'
    })

    # ── Forecast chart ──
    cutoff = series.index[-1]
    fut    = fc[fc['ds'] > cutoff]

    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(
        x=series.index, y=series.values,
        name='Historical', line=dict(color='steelblue')))
    fig2.add_trace(go.Scatter(
        x=pd.concat([fut['ds'], fut['ds'][::-1]]),
        y=pd.concat([fut['yhat_upper'], fut['yhat_lower'][::-1]]),
        fill='toself', fillcolor='rgba(255,99,71,0.15)',
        line=dict(color='rgba(0,0,0,0)'), name='95% CI'))
    fig2.add_trace(go.Scatter(
        x=fut['ds'], y=fut['yhat'],
        name='Forecast', line=dict(color='tomato', width=2.5)))
    fig2.add_vline(
        x=cutoff.timestamp() * 1000,
        line_dash='dash', line_color='gray',
        annotation_text='Forecast start')
    fig2.update_layout(
        title=f"{grp}  |  MAPE: {round(mape,1) if mape else 'N/A'}%",
        xaxis_title='Date', yaxis_title='Sales',
        template='plotly_white',
        legend=dict(orientation='h'))
    st.plotly_chart(fig2, use_container_width=True)

# ── 6. Metrics table ────────────────────────────────────────
st.subheader("Evaluation Metrics (last 12 months as test set)")
if metrics_list:
    st.dataframe(pd.DataFrame(metrics_list), use_container_width=True)
