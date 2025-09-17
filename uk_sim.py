# app.py
# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from prophet import Prophet
from warnings import filterwarnings
filterwarnings("ignore")

# =========================
# 1) 기본 설정
# =========================
FREQ_MINUTES = 15
STEPS_PER_DAY = 96
HORIZON = 3 * 24 * (60 // FREQ_MINUTES)   # 3일 예측 = 288
TRAIN_DAYS_LIST = [1, 2, 3]

PARAM_SETS = [
    {"trend": "add", "damped_trend": True,  "seasonal": "add", "sp": STEPS_PER_DAY},
    {"trend": "add", "damped_trend": False, "seasonal": "add", "sp": STEPS_PER_DAY},
    {"trend": "mul", "damped_trend": True,  "seasonal": "add", "sp": STEPS_PER_DAY},
    {"trend": "mul", "damped_trend": False, "seasonal": "add", "sp": STEPS_PER_DAY},
    {"trend": "add", "damped_trend": True,  "seasonal": "mul", "sp": STEPS_PER_DAY},
    {"trend": "add", "damped_trend": False, "seasonal": "mul", "sp": STEPS_PER_DAY},
    {"trend": "mul", "damped_trend": True,  "seasonal": "mul", "sp": STEPS_PER_DAY},
    {"trend": "mul", "damped_trend": False, "seasonal": "mul", "sp": STEPS_PER_DAY},
    {"trend": "add", "damped_trend": True,  "seasonal": "add", "sp": 48},
    {"trend": "add", "damped_trend": True,  "seasonal": None,   "sp": None},
]

# =========================
# 2) 유틸
# =========================
def sigma_band_from_residuals(y_true_train, y_fit_train, y_pred_test, z=3.0):
    resid = (y_true_train - y_fit_train).dropna()
    sigma = resid.std(ddof=1)
    low  = y_pred_test - z*sigma
    high = y_pred_test + z*sigma
    return low, high, sigma

def fit_and_forecast(ts_series, params, horizon, now_time):
    try:
        model = ExponentialSmoothing(
            ts_series,
            trend=params["trend"],
            damped_trend=params["damped_trend"],
            seasonal=params["seasonal"],
            seasonal_periods=params["sp"],
            initialization_method="estimated"
        )
        fit_res = model.fit(optimized=True)
    except Exception:
        model = ExponentialSmoothing(
            ts_series, trend="add", damped_trend=True,
            seasonal=None, initialization_method="estimated"
        )
        fit_res = model.fit(optimized=True)

    fc = fit_res.forecast(horizon)
    fc.index = pd.date_range(
        now_time + pd.Timedelta(minutes=FREQ_MINUTES),
        periods=horizon, freq=f"{FREQ_MINUTES}min"
    )
    return fc

# =========================
# 3) UI
# =========================
import base64
from PIL import Image

logo_path = "k-water ai lab.jpg"
with open(logo_path, "rb") as f:
    logo_data = f.read()
logo_base64 = base64.b64encode(logo_data).decode()

st.markdown(
    f"""
    <div style="display:flex; align-items:center;">
        <img src="data:image/png;base64,{logo_base64}" width="150" style="margin-right:10px;">
        <h1 style="margin:0;">Pressure Data: Exploratory Analysis, Forecasting & Anomaly Detection</h1>
    </div>
    """,
    unsafe_allow_html=True
)

# 기본 데이터 사용 여부
use_default = st.checkbox("Use default data (uk_data.csv)", value=True)

uploaded_file = None
if use_default:
    try:
        uploaded_file = "uk_data.csv"
        st.info("📂 Using default file: uk_data.csv")
    except Exception as e:
        st.error(f"Default file not found: {e}")
else:
    uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    df.columns = df.columns.str.strip()

    # === 기본 컬럼 선택 로직 ===
    time_default = 0
    value_default = 1

    # "Date Of Read" 컬럼이 있으면 무조건 Time 기본값으로
    if "Date Of Read" in df.columns:
        time_default = df.columns.get_loc("Date Of Read")

    # "Validated Value" 포함된 컬럼 자동 선택
    for col in df.columns:
        if "validated value" in col.lower():
            value_default = df.columns.get_loc(col)

    TIME_COL  = st.selectbox("Select Time Column", options=df.columns, index=time_default, key="timecol")
    VALUE_COL = st.selectbox("Select Value Column", options=df.columns, index=value_default, key="valuecol")

    # 전처리
    df[TIME_COL]  = pd.to_datetime(df[TIME_COL], dayfirst=True, errors="coerce")
    df[VALUE_COL] = pd.to_numeric(df[VALUE_COL], errors="coerce")
    df = df.dropna(subset=[TIME_COL, VALUE_COL]).sort_values(TIME_COL).set_index(TIME_COL)

    st.write("Data Preview", df.head())

    # === 이하 기존 분석/시각화 코드 동일 ===
    # (EDA, Holt-Winters, Prophet, Ensemble, Anomaly Detection...)


    st.write("Data Preview", df.head())

    # 2) 분석 옵션 + 실행 버튼
    st.markdown("---")
    colA, colB = st.columns(2)
    with colA:
        z = st.slider("Anomaly threshold (Z-sigma)", 1.0, 6.0, 3.0, 0.1)
    with colB:
        pass

    col1, col2, col3 = st.columns([1,2,1])
    with col2:
        run = st.button("🚀 **Run Analysis**", use_container_width=True)

    if run:
        with st.spinner("Running analysis..."):
            # === EDA ===
            st.subheader("📊 Exploratory Data Analysis")
            fig, axes = plt.subplots(2, 2, figsize=(14, 8))
            axes[0,0].plot(df.index, df[VALUE_COL]); axes[0,0].set_title("Full Series")
            axes[0,1].boxplot(df[VALUE_COL].values, showmeans=True); axes[0,1].set_title("Overall Boxplot")
            hours = range(0,24)
            hour_data = [df.loc[df.index.hour==h, VALUE_COL].values for h in hours]
            axes[1,0].boxplot(hour_data, labels=hours, showmeans=True); axes[1,0].set_title("Hour-of-Day Boxplot")
            dow = df.index.dayofweek; weekday_names=["Mon","Tue","Wed","Thu","Fri","Sat","Sun"]
            dow_data = [df.loc[dow==k, VALUE_COL].values for k in range(7)]
            axes[1,1].boxplot(dow_data, labels=weekday_names, showmeans=True); axes[1,1].set_title("Day-of-Week Boxplot")
            plt.tight_layout()
            st.pyplot(fig)

            # === Train/Test split (6일/1일) ===
            H = STEPS_PER_DAY
            train = df.iloc[:-H][VALUE_COL].copy()
            test  = df.iloc[-H:][VALUE_COL].copy()
            now_time = df.index.max()

            # === Holt-Winters ===
            hw_fit = ExponentialSmoothing(train, trend="add", seasonal="add",
                                          seasonal_periods=STEPS_PER_DAY).fit()
            hw_pred   = hw_fit.forecast(H)
            hw_fit_in = hw_fit.fittedvalues.reindex(train.index)

            # === Prophet ===
            train_p = train.reset_index().rename(columns={TIME_COL:"ds", VALUE_COL:"y"})
            m = Prophet(daily_seasonality=True, weekly_seasonality=True, yearly_seasonality=False)
            m.fit(train_p)
            future = m.make_future_dataframe(periods=H, freq="15min", include_history=False)
            fcst   = m.predict(future)
            prophet_pred = pd.Series(fcst["yhat"].values, index=pd.to_datetime(fcst["ds"].values))
            fit_all = m.predict(train_p[["ds"]])
            prophet_fit_in = pd.Series(fit_all["yhat"].values, index=train.index)

            # === HW Ensemble (+3 days) ===
            st.subheader("📈 Holt-Winters Ensemble Forecast (+3 days)")
            all_forecasts = []
            for days in TRAIN_DAYS_LIST:
                start_time = now_time - pd.Timedelta(days=days) + pd.Timedelta(minutes=FREQ_MINUTES)
                ts = df.loc[df.index >= start_time, VALUE_COL].asfreq(f"{FREQ_MINUTES}min").interpolate(limit_direction="both")
                for i, ps in enumerate(PARAM_SETS):
                    fc = fit_and_forecast(ts, ps, horizon=HORIZON, now_time=now_time)
                    temp = pd.DataFrame({"timestamp": fc.index, "predicted": fc.values,
                                         "train_days": days, "model_id": f"{days}d_{i}"})
                    all_forecasts.append(temp)
            all_forecasts = pd.concat(all_forecasts, ignore_index=True)

            q_summary = (all_forecasts
                         .groupby("timestamp")["predicted"]
                         .quantile([0.25, 0.5, 0.75])
                         .unstack()
                         .rename(columns={0.25: "q25", 0.5: "median", 0.75: "q75"}))

            fig, ax = plt.subplots(figsize=(14,6))
            ax.plot(df.index, df[VALUE_COL], label="Actual", color="blue")
            for mid in all_forecasts["model_id"].unique():
                sub = all_forecasts[all_forecasts["model_id"]==mid]
                ax.plot(sub["timestamp"], sub["predicted"], color="gray", alpha=0.2, linewidth=0.8)
            ax.fill_between(q_summary.index, q_summary["q25"], q_summary["q75"], color="red", alpha=0.3, label="Q25–Q75")
            ax.plot(q_summary.index, q_summary["median"], "r:", label="Median Forecast")
            ax.axvline(now_time, linestyle=":", color="red")
            ax.set_title("HW Ensemble Forecast (+3 days)")
            ax.legend(); st.pyplot(fig)

            # === 단일모델 비교 ===
            st.subheader("🔮 Holt-Winters vs Prophet (Last Day Forecast)")
            fig, ax = plt.subplots(figsize=(12,5))
            ax.plot(test.index, test.values, label="Test (True)", color="black")
            ax.plot(hw_pred.index, hw_pred.values, "--", label="Holt-Winters")
            ax.plot(prophet_pred.index, prophet_pred.values, "--", label="Prophet")
            ax.legend(); ax.set_title("Forecast Comparison on Test (1 day)")
            st.pyplot(fig)

            # === ±Zσ Anomaly Detection (표로 요약) ===
            st.subheader(f"⚠️ ±{z}σ Anomaly Detection")
            for name, fit_in, pred in [("Holt-Winters", hw_fit_in, hw_pred),
                                       ("Prophet", prophet_fit_in, prophet_pred)]:
                low, high, sigma = sigma_band_from_residuals(train, fit_in, pred, z=float(z))
                anom = (test < low) | (test > high)
                n_anom = int(anom.sum())

                fig, ax = plt.subplots(figsize=(12,5))
                ax.fill_between(pred.index, low, high, alpha=0.3, label=f"±{z}σ band")
                ax.plot(test.index, test.values, label="Test", color="black")
                ax.plot(pred.index, pred.values, "--", label=f"{name} Forecast")
                ax.scatter(test.index[anom], test[anom], color="red", marker="x", label="Anomaly")
                ax.set_title(f"{name} – ±{z}σ band (σ≈{sigma:.3f})")
                ax.legend(); st.pyplot(fig)

                if n_anom > 0:
                    anom_idx = test.index[anom]
                    anom_df = pd.DataFrame({
                        "Timestamp": anom_idx.strftime("%Y-%m-%d %H:%M"),
                        "Actual": test[anom].values,
                        "Forecast": pred.reindex(anom_idx).values,
                        f"Lower (±{z}σ)": low.reindex(anom_idx).values,
                        f"Upper (±{z}σ)": high.reindex(anom_idx).values,
                    }).reset_index(drop=True)

                    st.error(f"🚨 ABNORMAL — {name}: {n_anom} anomalies detected")
                    st.dataframe(anom_df, use_container_width=True)
                else:
                    st.success(f"✅ No anomalies detected in {name}")

        st.success("✅ Analysis complete! You can now explore the result graphs above.")


