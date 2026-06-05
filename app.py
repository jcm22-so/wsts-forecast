import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from prophet import Prophet
from statsmodels.tsa.statespace.sarimax import SARIMAX

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

# Model selector
st.sidebar.markdown("---")
st.sidebar.header("Models")
run_prophet = st.sidebar.checkbox("Prophet", value=True)
run_sarima  = st.sidebar.checkbox("SARIMA",  value=True)

if not selected:
    st.warning("Please select at least one option from the sidebar.")
    st.stop()

if not run_prophet and not run_sarima:
    st.warning("Please select at least one model from the sidebar.")
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

# ── 5. Forecast per group ───────────────────────────────────
metrics_list = []

for grp in selected:
    st.markdown(f"---\n### {grp}")

    series = (df_filtered[df_filtered[group_col] == grp]
              .groupby('Date')['Value'].sum()
              .rename('Value')
              .asfreq('MS'))

    if len(series) < 24:
        st.warning(f"'{grp}' has less than 24 months of data — skipping.")
        continue

    train = series.iloc[:-12]
    test  = series.iloc[-12:]
    cutoff = series.index[-1]

    cols = st.columns(2)

    # ════════════════════════════════
    # PROPHET
    # ════════════════════════════════
    if run_prophet:
        with cols[0]:
            st.markdown("#### 🔮 Prophet")
            with st.spinner("Fitting Prophet..."):
                df_p = train.reset_index().rename(columns={'Date':'ds','Value':'y'})
                m = Prophet(yearly_seasonality=True, weekly_seasonality=False,
                            daily_seasonality=False, interval_width=0.95)
                m.fit(df_p)
                future = m.make_future_dataframe(periods=12 + forecast_months, freq='MS')
                fc_p   = m.predict(future)

            pred_test_p = fc_p[fc_p['ds'].isin(test.index)]['yhat'].values
            if len(pred_test_p) == len(test):
                mad_p  = float(np.mean(np.abs(test.values - pred_test_p)))
                mse_p  = float(np.mean((test.values - pred_test_p) ** 2))
                mape_p = float(np.mean(np.abs((test.values - pred_test_p) / test.values)) * 100)
            else:
                mad_p, mse_p, mape_p = None, None, None

            fut_p = fc_p[fc_p['ds'] > cutoff]
            fig_p = go.Figure()
            fig_p.add_trace(go.Scatter(x=series.index, y=series.values,
                                       name='Historical', line=dict(color='steelblue')))
            fig_p.add_trace(go.Scatter(
                x=pd.concat([fut_p['ds'], fut_p['ds'][::-1]]),
                y=pd.concat([fut_p['yhat_upper'], fut_p['yhat_lower'][::-1]]),
                fill='toself', fillcolor='rgba(255,99,71,0.15)',
                line=dict(color='rgba(0,0,0,0)'), name='95% CI'))
            fig_p.add_trace(go.Scatter(x=fut_p['ds'], y=fut_p['yhat'],
                                       name='Forecast', line=dict(color='tomato', width=2.5)))
            fig_p.add_vline(x=cutoff.timestamp()*1000, line_dash='dash',
                            line_color='gray', annotation_text='Forecast start')
            fig_p.update_layout(
                title=f"Prophet — MAPE: {round(mape_p,1) if mape_p else 'N/A'}%",
                xaxis_title='Date', yaxis_title='Sales',
                template='plotly_white', legend=dict(orientation='h'))
            st.plotly_chart(fig_p, use_container_width=True)

            metrics_list.append({
                group_col: grp, 'Model': 'Prophet',
                'MAD':    round(mad_p,  0) if mad_p  is not None else 'N/A',
                'MSE':    round(mse_p,  0) if mse_p  is not None else 'N/A',
                'MAPE_%': round(mape_p, 2) if mape_p is not None else 'N/A'
            })

    # ════════════════════════════════
    # SARIMA
    # ════════════════════════════════
    if run_sarima:
        with cols[1] if run_prophet else cols[0]:
            st.markdown("#### 📈 SARIMA")
            with st.spinner("Fitting SARIMA..."):
                model  = SARIMAX(train, order=(1,1,1), seasonal_order=(1,1,1,12),
                                 enforce_stationarity=False, enforce_invertibility=False)
                result = model.fit(disp=False)
                fc_s   = result.get_forecast(steps=12 + forecast_months)
                fc_mean = fc_s.predicted_mean
                fc_ci   = fc_s.conf_int()

            pred_test_s = fc_mean.iloc[:12].values
            if len(pred_test_s) == len(test):
                mad_s  = float(np.mean(np.abs(test.values - pred_test_s)))
                mse_s  = float(np.mean((test.values - pred_test_s) ** 2))
                mape_s = float(np.mean(np.abs((test.values - pred_test_s) / test.values)) * 100)
            else:
                mad_s, mse_s, mape_s = None, None, None

            fut_mean = fc_mean.iloc[12:]
            fut_ci   = fc_ci.iloc[12:]

            fig_s = go.Figure()
            fig_s.add_trace(go.Scatter(x=series.index, y=series.values,
                                       name='Historical', line=dict(color='steelblue')))
            fig_s.add_trace(go.Scatter(
                x=pd.concat([fut_mean.index.to_series(), fut_mean.index.to_series()[::-1]]),
                y=pd.concat([fut_ci.iloc[:,1], fut_ci.iloc[:,0][::-1]]),
                fill='toself', fillcolor='rgba(147,112,219,0.15)',
                line=dict(color='rgba(0,0,0,0)'), name='95% CI'))
            fig_s.add_trace(go.Scatter(x=fut_mean.index, y=fut_mean.values,
                                       name='Forecast', line=dict(color='mediumpurple', width=2.5)))
            fig_s.add_vline(x=cutoff.timestamp()*1000, line_dash='dash',
                            line_color='gray', annotation_text='Forecast start')
            fig_s.update_layout(
                title=f"SARIMA — MAPE: {round(mape_s,1) if mape_s else 'N/A'}%",
                xaxis_title='Date', yaxis_title='Sales',
                template='plotly_white', legend=dict(orientation='h'))
            st.plotly_chart(fig_s, use_container_width=True)

            metrics_list.append({
                group_col: grp, 'Model': 'SARIMA',
                'MAD':    round(mad_s,  0) if mad_s  is not None else 'N/A',
                'MSE':    round(mse_s,  0) if mse_s  is not None else 'N/A',
                'MAPE_%': round(mape_s, 2) if mape_s is not None else 'N/A'
            })

# ── 6. Metrics table ────────────────────────────────────────
st.markdown("---")
st.subheader("📊 Evaluation Metrics — Prophet vs SARIMA")
if metrics_list:
    metrics_df = pd.DataFrame(metrics_list)
    st.dataframe(metrics_df, use_container_width=True)

    # MAPE comparison bar chart
    mape_df = metrics_df[metrics_df['MAPE_%'] != 'N/A'].copy()
    mape_df['MAPE_%'] = mape_df['MAPE_%'].astype(float)

    fig_m = go.Figure()
    for model_name, color in [('Prophet','tomato'), ('SARIMA','mediumpurple')]:
        sub = mape_df[mape_df['Model'] == model_name]
        if not sub.empty:
            fig_m.add_trace(go.Bar(x=sub[group_col], y=sub['MAPE_%'],
                                   name=model_name, marker_color=color))
    fig_m.add_hline(y=10, line_dash='dash', line_color='green',
                    annotation_text='10% — Excellent')
    fig_m.add_hline(y=20, line_dash='dash', line_color='orange',
                    annotation_text='20% — Acceptable')
    fig_m.update_layout(
        title='MAPE Comparison — Prophet vs SARIMA',
        xaxis_title=view_by, yaxis_title='MAPE (%)',
        barmode='group', template='plotly_white',
        legend=dict(orientation='h'))
    st.plotly_chart(fig_m, use_container_width=True)

    # Best model per group
    st.subheader("🏆 Best Model per Group (lowest MAPE)")
    best = (mape_df.sort_values('MAPE_%')
                   .groupby(group_col)
                   .first()
                   .reset_index()[[group_col, 'Model', 'MAD', 'MSE', 'MAPE_%']])
    st.dataframe(best, use_container_width=True)
