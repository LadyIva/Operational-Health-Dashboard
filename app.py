import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
import plotly.graph_objects as go
import plotly.express as px
import time
import os

# --- Configuration ---
st.set_page_config(layout="wide", page_title="S.I.L.K.E. Predictive Maintenance Demo")

# Add the st.logo function call at the top of your script
st.logo("silke_logo_transparent.png", size="large")

DATA_POINT_INTERVAL = 1.0
file_path = "maize_mill_simulated_sensor_data.csv"


# --- 1. Load Data and Initialize Models (Runs only once) ---
@st.cache_data
def load_data(file_path):
    try:
        if not os.path.exists(file_path):
            st.error(f"Error: File not found at {file_path}. Please check the path.")
            st.stop()

        df = pd.read_csv(file_path, parse_dates=["Timestamp"])

        if df.empty:
            return pd.DataFrame(
                columns=[
                    "Timestamp",
                    "Power_kW",
                    "Amperage",
                    "Vibration",
                    "Temperature",
                ]
            )

        required_columns = [
            "Timestamp",
            "Power_kW",
            "Amperage",
            "Vibration",
            "Temperature",
        ]
        if not all(col in df.columns for col in required_columns):
            st.error("Error: The CSV file is missing one of the required columns.")
            st.error(f"Required columns: {required_columns}")
            st.error(f"Found columns: {list(df.columns)}")
            st.stop()

        df = df.set_index("Timestamp").sort_index()

        window_size = "12h"
        df["Power_kW_RollingMean"] = df["Power_kW"].rolling(window=window_size).mean()
        df["Power_kW_RollingStd"] = df["Power_kW"].rolling(window=window_size).std()
        df["Amperage_RollingMean"] = df["Amperage"].rolling(window=window_size).mean()
        df["Amperage_RollingStd"] = df["Amperage"].rolling(window=window_size).std()

        return df
    except FileNotFoundError:
        st.error(f"Error: File not found at {file_path}. Please check the path.")
        st.stop()
    except pd.errors.EmptyDataError:
        st.error("The CSV file is empty.")
        st.stop()


@st.cache_resource
def train_isolation_forest(df_initial):
    if df_initial.empty:
        return None, None

    features_for_ml = ["Power_kW", "Amperage", "Vibration", "Temperature"]
    X_train = df_initial[features_for_ml].dropna()
    if X_train.empty or len(X_train) < 2:
        return None, None
    model = IsolationForest(contamination=0.01, random_state=42)
    model.fit(X_train.values)
    return model, features_for_ml


# Global data and model initialization
full_data_df = load_data(file_path)

try:
    iso_forest_model, ml_features = train_isolation_forest(full_data_df.head(1000))
except Exception as e:
    st.error(f"An error occurred during model training: {e}")
    st.warning(
        "Model training failed. This may be due to insufficient data or inconsistencies."
    )
    st.stop()


# --- 2. Anomaly Detection Logic (as functions) ---
def check_rule_based_anomalies(row):
    """Applies rule-based anomaly detection to a single data row."""
    rule_power_threshold = 600
    rule_temp_threshold = 70.0
    rule_vibration_threshold = 6.5
    rule_temp_gradual_threshold = 75.0

    anomaly_rule_1 = (row["Power_kW"] > rule_power_threshold) and (
        row["Temperature"] > rule_temp_threshold
    )

    try:
        if not pd.isna(row["Power_kW_RollingMean"]):
            rolling_mean = row["Power_kW_RollingMean"]
            rolling_std = row["Power_kW_RollingStd"]
            rolling_mean_amp = row["Amperage_RollingMean"]
            rolling_std_amp = row["Amperage_RollingStd"]

            anomaly_rule_2 = (row["Power_kW"] > (rolling_mean + 3 * rolling_std)) or (
                row["Amperage"] > (rolling_mean_amp + 3 * rolling_std_amp)
            )
        else:
            anomaly_rule_2 = False
    except KeyError:
        anomaly_rule_2 = False

    anomaly_rule_3 = (row["Vibration"] > rule_vibration_threshold) and (
        row["Temperature"] > rule_temp_gradual_threshold
    )

    is_rule_anomaly = anomaly_rule_1 or anomaly_rule_2 or anomaly_rule_3
    reasoning = ""
    if anomaly_rule_1:
        reasoning += "High Power & Temp: Potential inefficiency/overload. "
    if anomaly_rule_2:
        reasoning += "Sudden Power/Amp Spike. "
    if anomaly_rule_3:
        reasoning += "Gradual Vibe/Temp Increase: Possible mechanical wear. "

    return is_rule_anomaly, reasoning.strip()


def check_ml_anomaly(row, model, features):
    """Applies the Isolation Forest model to a single data row."""
    if model is None or features is None or any(pd.isna(row[features])):
        return False, 0.0

    data_point = np.array([row[features].values])
    ml_prediction = model.predict(data_point)[0]
    ml_score = model.decision_function(data_point)[0]
    is_ml_anomaly = ml_prediction == -1

    return is_ml_anomaly, ml_score


# --- 3. Streamlit UI Rendering and Simulation Logic ---
st.title("S.I.L.K.E. Predictive Maintenance Demo")
st.write("Live dashboard displaying sensor data and detecting anomalies in real-time.")

# --- NEW: Sidebar content ---
with st.sidebar:
    st.header("About This Demo")
    st.info(
        "💡 This dashboard showcases the potential of our AIoT solutions. It uses simulated data to demonstrate how we can predict equipment failure before it happens, ensuring seamless operations."
    )

    st.markdown("---")
    st.header("The Value Proposition")

    st.markdown(
        """
    Our solution helps you shift from reactive maintenance to proactive prevention of issues by providing:

    -   **Predictive Maintenance & Uptime:** Our AI models identify anomalies before they cause critical failures, reducing unexpected downtime and extending equipment lifespan.
    -   **Optimized Operations:** Timely adjustments and data-driven insights help you improve efficiency and minimize energy waste.
    -   **Actionable Intelligence:** We provide a unified view of your entire facility’s operational health, turning complex data into actionable insights for informed decisions and improved safety.
    """
    )

if "current_df" not in st.session_state:
    initial_row = full_data_df.head(1).copy()
    initial_row["Is_Rule_Anomaly"] = False
    initial_row["Is_ML_Anomaly"] = False
    initial_row["Anomaly_Reasoning"] = ""
    initial_row["ML_Anomaly_Score"] = 0.0
    st.session_state.current_df = initial_row
    st.session_state.current_row_index = 1
    st.session_state.anomaly_count = 0

# Create empty placeholders for all dynamic UI elements
kpi_placeholder = st.empty()
alert_placeholder = st.empty()
chart_placeholder = st.empty()


def update_dashboard(kpi_ph, alert_ph, chart_ph, current_df, anomaly_count):
    """Function to update all dynamic UI elements at once."""
    if current_df.empty:
        return

    latest_row = current_df.iloc[-1]
    with kpi_ph.container():
        st.subheader("Key Performance Indicators")
        col_kpi1, col_kpi2, col_kpi3, col_kpi4 = st.columns(4)
        col_kpi1.metric("Latest Power (kW)", f"{latest_row['Power_kW']:.2f}")
        col_kpi2.metric("Latest Vibration", f"{latest_row['Vibration']:.2f}")
        col_kpi3.metric("Latest Temperature (°C)", f"{latest_row['Temperature']:.2f}")
        col_kpi4.metric("Total Anomalies", anomaly_count)

    with alert_ph.container():
        if anomaly_count > 0:
            last_anomaly = current_df[
                current_df["Is_Rule_Anomaly"] | current_df["Is_ML_Anomaly"]
            ].iloc[-1]
            st.error(
                f"⚠️ Anomaly Detected at {last_anomaly.name.strftime('%Y-%m-%d %H:%M:%S')}!"
            )
            st.markdown(
                f"**Reasoning:** {last_anomaly['Anomaly_Reasoning'] if last_anomaly['Anomaly_Reasoning'] else 'ML Detected: Uncategorized Anomaly.'}"
            )

            with st.expander("Show Detailed Anomaly Report"):
                anomalies_to_show = current_df[
                    current_df["Is_Rule_Anomaly"] | current_df["Is_ML_Anomaly"]
                ]
                st.dataframe(
                    anomalies_to_show[
                        [
                            "Power_kW",
                            "Vibration",
                            "Temperature",
                            "Anomaly_Reasoning",
                            "Is_ML_Anomaly",
                            "ML_Anomaly_Score",
                        ]
                    ],
                    use_container_width=True,
                )

                with st.expander("Show Latest Anomaly Chart"):
                    df_anomaly_window = current_df.loc[
                        (current_df.index >= last_anomaly.name - pd.Timedelta(hours=2))
                        & (
                            current_df.index
                            <= last_anomaly.name + pd.Timedelta(hours=2)
                        )
                    ]
                    fig_anomaly = px.line(
                        df_anomaly_window,
                        x=df_anomaly_window.index,
                        y=["Power_kW", "Vibration", "Temperature"],
                        title=f"Sensor Readings Around Anomaly at {last_anomaly.name.strftime('%Y-%m-%d %H:%M:%S')}",
                    )

                    fig_anomaly.add_trace(
                        go.Scatter(
                            x=[last_anomaly.name],
                            y=[last_anomaly["Power_kW"]],
                            mode="markers",
                            marker=dict(color="red", size=15, symbol="x"),
                            name="Anomaly Point",
                        )
                    )

                    st.plotly_chart(
                        fig_anomaly,
                        use_container_width=True,
                        key=f"anomaly_chart_{last_anomaly.name.isoformat()}_{st.session_state.current_row_index}",
                    )

        else:
            st.success("✅ System Operating Normally.")

    with chart_ph.container():
        st.subheader("Real-time Sensor Data Monitoring")
        fig_main = px.line(
            current_df,
            x=current_df.index,
            y=["Power_kW", "Vibration", "Temperature"],
            labels={"value": "Value", "Timestamp": "Time"},
            title="Live Sensor Data Stream",
        )
        fig_main.update_layout(height=500, xaxis_title="Timestamp")
        st.plotly_chart(
            fig_main,
            use_container_width=True,
            key=f"main_chart_{st.session_state.current_row_index}",
        )


# The continuous simulation loop
while st.session_state.current_row_index < len(full_data_df):
    try:
        next_row = full_data_df.iloc[
            st.session_state.current_row_index : st.session_state.current_row_index + 1
        ].copy()

        is_rule_anomaly, rule_reasoning = check_rule_based_anomalies(next_row.iloc[0])
        is_ml_anomaly, ml_score = check_ml_anomaly(
            next_row.iloc[0], iso_forest_model, ml_features
        )

        next_row.loc[:, "Is_Rule_Anomaly"] = is_rule_anomaly
        next_row.loc[:, "Is_ML_Anomaly"] = is_ml_anomaly
        next_row.loc[:, "Anomaly_Reasoning"] = rule_reasoning
        next_row.loc[:, "ML_Anomaly_Score"] = ml_score

        st.session_state.current_df = pd.concat([st.session_state.current_df, next_row])
        st.session_state.current_row_index += 1
        if is_rule_anomaly or is_ml_anomaly:
            st.session_state.anomaly_count += 1

        # Call the new function to update all UI elements
        update_dashboard(
            kpi_placeholder,
            alert_placeholder,
            chart_placeholder,
            st.session_state.current_df,
            st.session_state.anomaly_count,
        )

        time.sleep(DATA_POINT_INTERVAL)

    except Exception as e:
        st.error(
            f"⚠️ Error: The simulation crashed while processing row {st.session_state.current_row_index}."
        )
        st.error(f"**Details:** {e}")
        st.warning(
            "The simulation has stopped. Please check your data at this row for any inconsistencies (e.g., missing values, incorrect data types)."
        )
        st.stop()

# Final status message after the loop
if st.session_state.current_row_index >= len(full_data_df):
    st.info("End of simulation. All data has been processed.")
