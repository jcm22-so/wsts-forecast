import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from prophet import Prophet
from statsmodels.tsa.statespace.sarimax import SARIMAX
from datetime import datetime

st.set_page_config(page_title="WSTS Semiconductor Forecast", layout="wide")

# ════════════════════════════════════════════════════════════
# ACTIVITY LOG — stored in session_state across users
# ════════════════════════════════════════════════════════════
if "activity_log" not in st.session_state:
    st.session_state.activity_log = []   # list of dicts

def log_activity(username, action):
    st.session_state.activity_log.append({
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "user":      username,
        "action":    action
    })

# ════════════════════════════════════════════════════════════
# LOGIN
# ════════════════════════════════════════════════════════════
def check_login():
    if "logged_in" not in st.session_state:
        st.session_state.logged_in = False
        st.session_state.username  = ""
        st.session_state.role      = ""

    if not st.session_state.logged_in:
        col1, col2, col3 = st.columns([1, 1.2, 1])
        with col2:
            st.markdown("<br><br>", unsafe_allow_html=True)
            st.title("🔐 WSTS Forecast")
            st.markdown("### Please log in to continue")
            st.markdown("---")

            username = st.text_input("Username")
            password = st.text_input("Password", type="password")

            if st.button("Login", use_container_width=True):
                credentials = st.secrets.get("credentials", {})
                roles       = st.secrets.get("roles", {})
                if username in credentials and credentials[username] == password:
                    st.session_state.logged_in = True
                    st.session_state.username  = username
                    st.session_state.role      = roles.get(username, "viewer")
                    log_activity(username, "Logged in")
                    st.rerun()
                else:
                    st.error("❌ Incorrect username or password")
        st.stop()

check_login()

# ── Sidebar user info & logout ───────────────────────────────
role     = st.session_state.role
username = st.session_state.username

st.sidebar.markdown(f"👤 **{username}** `({role})`")
if st.sidebar.button("Logout"):
    log_activity(username, "Logged out")
    st.session_state.logged_in = False
    st.session_state.username  = ""
    st.session_state.role      = ""
    st.rerun()

st.sidebar.markdown("---")

# ════════════════════════════════════════════════════════════
# NAVIGATION — admin sees extra Users panel
# ════════════════════════════════════════════════════════════
st.sidebar.title("📂 Navigation")

if role == "admin":
    pages = ["🏠 Introduction", "📊 Power BI Dashboards", "🔮 Predictive Models", "👥 User Activity"]
else:
    pages = ["🏠 Introduction", "📊 Power BI Dashboards", "🔮 Predictive Models"]

page = st.sidebar.radio("Go to", pages)

# Log page visits
if "last_page" not in st.session_state:
    st.session_state.last_page = ""
if st.session_state.last_page != page:
    log_activity(username, f"Visited page: {page}")
    st.session_state.last_page = page

# ════════════════════════════════════════════════════════════
# PAGE 1 — INTRODUCTION
# ════════════════════════════════════════════════════════════
if page == "🏠 Introduction":
    st.title("🏠 Semiconductor Sales Analysis")
    st.markdown("---")
    st.markdown("""
    ## About this project
    This application analyzes global semiconductor sales data from the **WSTS (World Semiconductor Trade Statistics)** dataset.
    It provides time series analysis and forecasting by **Category** and **Region** using machine learning models.
    """)

    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    col1.metric("📦 Dataset", "WSTS")
    col2.metric("📅 Frequency", "Monthly")
    col3.metric("🌍 Coverage", "Global")

    st.markdown("---")
    st.markdown("## 📋 Dataset Structure")
    st.markdown("""
    | Column | Description |
    |--------|-------------|
    | **Region** | Geographic region of sales |
    | **Category** | Product category (e.g. Total Analog, Total Logic) |
    | **Year** | Year of the record |
    | **Month** | Month of the record |
    | **Value** | Sales value in USD |
    """)

    st.markdown("---")
    st.markdown("## 🔍 What you can do here")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.info("### 📊 Power BI Dashboards\nExplore interactive dashboards with sales trends, market share, and regional breakdowns.")
    with c2:
        st.success("### 🔮 Predictive Models\nRun Prophet and SARIMA forecasts by Category or Region with MAD, MSE, and MAPE metrics.")
    with c3:
        st.warning("### 🏆 Model Comparison\nCompare both models and identify which one predicts best for each group.")

    st.markdown("---")
    st.markdown("## 📈 Models Used")
    st.markdown("""
    **Prophet** — Developed by Meta. Automatically detects trends and yearly seasonality.
    Robust to missing data and outliers. Generates a 95% confidence interval.

    **SARIMA** — Classic time series model. Learns from past values of the series to predict future ones.
    The seasonal component captures patterns that repeat every 12 months.
    """)

    st.markdown("---")
    st.markdown("## 📐 Evaluation Metrics")
    st.markdown("""
    | Metric | Description |
    |--------|-------------|
    | **MAD** | Mean Absolute Deviation — average error in original units |
    | **MSE** | Mean Squared Error — penalizes large errors more heavily |
    | **MAPE** | Mean Absolute Percentage Error — error as a percentage |

    **MAPE interpretation:** < 10% Excellent · 10–20% Good · 20–50% Acceptable · > 50% Poor
    """)

# ════════════════════════════════════════════════════════════
# PAGE 2 — POWER BI DASHBOARDS
# ════════════════════════════════════════════════════════════
elif page == "📊 Power BI Dashboards":
    st.title("📊 Power BI Dashboards")
    st.markdown("---")

    import streamlit.components.v1 as components

    POWERBI_URL_1 = ""   https://app.powerbi.com/reportEmbed?reportId=4c7820fb-ef98-4363-a4d3-07fd437bfed2&autoAuth=true&ctid=f07b40ae-b60b-4e0f-bebe-afb42fc4dc69
    POWERBI_URL_2 = ""   # ← Paste a second dashboard URL here (optional)

    if not POWERBI_URL_1:
        st.info("""
        **How to embed your Power BI dashboard:**
        1. Open your report in Power BI Service (app.powerbi.com)
        2. Click **File → Embed report → Website or portal**
        3. Copy the embed URL
        4. Paste it in `app.py` where it says `POWERBI_URL_1 = ""`
        """)
        st.image("https://upload.wikimedia.org/wikipedia/commons/c/cf/New_Power_BI_Logo.svg", width=120)
        st.markdown("Your Power BI dashboards will appear here once you add the embed URLs.")
    else:
        st.markdown("### Dashboard 1")
        components.iframe(src=POWERBI_URL_1, width=1200, height=600, scrolling=True)
        if POWERBI_URL_2:
            st.markdown("---")
            st.markdown("### Dashboard 2")
            components.iframe(src=POWERBI_URL_2, width=1200, height=600, scrolling=True)

# ════════════════════════════════════════════════════════════
# PAGE 3 — PREDICTIVE MODELS
# ════════════════════════════════════════════════════════════
elif page == "🔮 Predictive Models":
    st.title("🔮 Predictive Models — Time Series & Forecast")
    st.markdown("---")

    uploaded = st.file_uploader("Upload WSTS.xlsx", type=["xlsx"])
    if uploaded is None:
        st.info("Please upload your WSTS.xlsx file to continue.")
        st.stop()

    log_activity(username, "Uploaded WSTS.xlsx")

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

    st.sidebar.markdown("---")
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

    df_filtered = df[df[group_col].isin(selected)].copy()

    st.subheader(f"Monthly Sales by {view_by}")
    ts = df_filtered.groupby(['Date', group_col])['Value'].sum().reset_index()
    fig = go.Figure()
    for grp in selected:
        sub = ts[ts[group_col] == grp]
        fig.add_trace(go.Scatter(x=sub['Date'], y=sub['Value'], mode='lines', name=grp))
    fig.update_layout(template='plotly_white', xaxis_title='Date', yaxis_title='Sales',
                      legend=dict(orientation='h'))
    st.plotly_chart(fig, use_container_width=True)

    metrics_list = []

    for grp in selected:
        st.markdown("---")
        st.markdown(f"## {grp}")

        series = (df_filtered[df_filtered[group_col] == grp]
                  .groupby('Date')['Value'].sum()
                  .rename('Value').asfreq('MS'))

        if len(series) < 24:
            st.warning(f"'{grp}' has less than 24 months of data — skipping.")
            continue

        train  = series.iloc[:-12]
        test   = series.iloc[-12:]
        cutoff = series.index[-1]

        if run_prophet:
            st.markdown("#### 🔮 Prophet")
            with st.spinner(f"Fitting Prophet for {grp}..."):
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

            log_activity(username, f"Ran Prophet forecast — {grp} ({forecast_months} months)")

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
                title=f"Prophet — {grp}  |  MAPE: {round(mape_p,1) if mape_p else 'N/A'}%",
                xaxis_title='Date', yaxis_title='Sales',
                template='plotly_white', legend=dict(orientation='h'))
            st.plotly_chart(fig_p, use_container_width=True)

            metrics_list.append({
                group_col: grp, 'Model': 'Prophet',
                'MAD':    round(mad_p,  0) if mad_p  is not None else 'N/A',
                'MSE':    round(mse_p,  0) if mse_p  is not None else 'N/A',
                'MAPE_%': round(mape_p, 2) if mape_p is not None else 'N/A'
            })

        if run_sarima:
            st.markdown("#### 📈 SARIMA")
            with st.spinner(f"Fitting SARIMA for {grp}..."):
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

            log_activity(username, f"Ran SARIMA forecast — {grp} ({forecast_months} months)")

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
                title=f"SARIMA — {grp}  |  MAPE: {round(mape_s,1) if mape_s else 'N/A'}%",
                xaxis_title='Date', yaxis_title='Sales',
                template='plotly_white', legend=dict(orientation='h'))
            st.plotly_chart(fig_s, use_container_width=True)

            metrics_list.append({
                group_col: grp, 'Model': 'SARIMA',
                'MAD':    round(mad_s,  0) if mad_s  is not None else 'N/A',
                'MSE':    round(mse_s,  0) if mse_s  is not None else 'N/A',
                'MAPE_%': round(mape_s, 2) if mape_s is not None else 'N/A'
            })

    st.markdown("---")
    st.subheader("📊 Evaluation Metrics — Prophet vs SARIMA")
    if metrics_list:
        metrics_df = pd.DataFrame(metrics_list)
        st.dataframe(metrics_df, use_container_width=True)

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

        st.subheader("🏆 Best Model per Group (lowest MAPE)")
        best = (mape_df.sort_values('MAPE_%')
                       .groupby(group_col).first()
                       .reset_index()[[group_col, 'Model', 'MAD', 'MSE', 'MAPE_%']])
        st.dataframe(best, use_container_width=True)

# ════════════════════════════════════════════════════════════
# PAGE 4 — USER ACTIVITY (admin only)
# ════════════════════════════════════════════════════════════
elif page == "👥 User Activity":
    st.title("👥 User Activity Panel")
    st.markdown("---")

    credentials = st.secrets.get("credentials", {})
    roles       = st.secrets.get("roles", {})

    # ── Registered users table ──
    st.subheader("📋 Registered Users")
    users_data = [{"Username": u, "Role": roles.get(u, "viewer")} for u in credentials]
    st.dataframe(pd.DataFrame(users_data), use_container_width=True)

    st.markdown("---")

    # ── Activity log ──
    st.subheader("📜 Activity Log")

    log = st.session_state.activity_log
    if not log:
        st.info("No activity recorded yet in this session.")
    else:
        log_df = pd.DataFrame(log)

        # Filter by user
        all_users = ["All"] + sorted(log_df['user'].unique().tolist())
        filter_user = st.selectbox("Filter by user", all_users)
        if filter_user != "All":
            log_df = log_df[log_df['user'] == filter_user]

        st.dataframe(log_df.sort_values('timestamp', ascending=False),
                     use_container_width=True)

        st.markdown("---")

        # ── Summary per user ──
        st.subheader("📊 Activity Summary per User")
        summary = (pd.DataFrame(st.session_state.activity_log)
                   .groupby('user')
                   .agg(
                       Total_Actions=('action', 'count'),
                       Last_Seen=('timestamp', 'max'),
                       Logins=('action', lambda x: (x == 'Logged in').sum())
                   )
                   .reset_index()
                   .rename(columns={'user':'User'}))
        st.dataframe(summary, use_container_width=True)

        # ── Actions bar chart ──
        fig_a = go.Figure()
        fig_a.add_trace(go.Bar(
            x=summary['User'],
            y=summary['Total_Actions'],
            marker_color='steelblue',
            text=summary['Total_Actions'],
            textposition='auto'
        ))
        fig_a.update_layout(
            title='Total Actions per User',
            xaxis_title='User', yaxis_title='Actions',
            template='plotly_white')
        st.plotly_chart(fig_a, use_container_width=True)

        # ── Download log ──
        st.markdown("---")
        csv = pd.DataFrame(st.session_state.activity_log).to_csv(index=False)
        st.download_button(
            label="⬇️ Download activity log as CSV",
            data=csv,
            file_name="activity_log.csv",
            mime="text/csv"
        )
