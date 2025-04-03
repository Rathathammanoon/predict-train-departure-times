import streamlit as st
import matplotlib.pyplot as plt
import warnings
from components.data_processor import load_and_preprocess_data
from components.model_trainer import train_model
from components.predictor import predict_single_input
from components.ui_components import render_train_input_form
import pickle
import os

warnings.filterwarnings('ignore')
plt.rcParams['font.family'] = 'Tahoma'

def main():
    st.set_page_config(page_title="ระบบทำนายการล่าช้าของรถไฟ", page_icon="🚂", layout="wide")

    st.title("🚂 ระบบทำนายการล่าช้าของรถไฟ")
    st.markdown("---")

    st.sidebar.title("เมนูหลัก")
    page = st.sidebar.radio(
        "เลือกหน้า",
        ["ฝึกโมเดล", "ทำนายการล่าช้า", "เกี่ยวกับระบบ"]
    )

    if page == "ฝึกโมเดล":
        render_training_page()
    elif page == "ทำนายการล่าช้า":
        render_prediction_page()
    elif page == "เกี่ยวกับระบบ":
        render_about_page()

def render_training_page():
    st.header("🧠 ฝึกโมเดลทำนายการล่าช้าของรถไฟ")
    uploaded_file = st.file_uploader("อัปโหลดไฟล์ Excel (.xlsx)", type=["xlsx"])

    if uploaded_file is not None:
        df = load_and_preprocess_data(uploaded_file)
        if df is not None:
            st.subheader("ข้อมูลตัวอย่าง")
            st.dataframe(df.head())

            default_features = [
                'Train_type',
                'Date',
                'Train_number',
                'Maximum_delay',
                'Outbound_trips_Return_trips',
                'Number_of_stations',
                'Number_of_junctions',
                'Railway_line',
                'Total_delay_time',
                'Departure_delay_origin',
                'Number_of_delayed_stations',
                'Time_period',
                'Maximum_train_speed',
                'Actual_departure_time_origin',
                'Actual_arrival_time_destination',
                'Scheduled_departure_time_origin',
                'Scheduled_arrival_time_destination',
                'Maximum_train_speed',
                'Distance'
            ]

            features = st.multiselect(
                "เลือก Features",
                df.columns.tolist(),
                default=list(set(default_features).intersection(set(df.columns.tolist())))
            )

            target = st.selectbox(
                "เลือก Target (ค่าที่ต้องการทำนาย)",
                df.columns.tolist(),
                index=df.columns.tolist().index("Arrival_delay_destination") if "Arrival_delay_destination" in df.columns else 0
            )

            if st.button("เริ่มฝึกโมเดล"):
                model, label_encoders, feature_cols = train_model(df, features, target)

def render_prediction_page():
    st.header("🔍 ทำนายการล่าช้าของรถไฟ")

    model_exists = os.path.exists('train_model.pkl') and os.path.exists('label_encoders.pkl')
    if not model_exists:
        st.warning("⚠️ ยังไม่มีโมเดลที่บันทึกไว้ กรุณาอัปโหลดโมเดลหรือไปที่หน้า 'ฝึกโมเดล' ก่อน")
        return

    try:
        with open('train_model.pkl', 'rb') as f:
            model = pickle.load(f)
        with open('label_encoders.pkl', 'rb') as f:
            label_encoders = pickle.load(f)

        st.subheader("กรอกข้อมูลเพื่อทำนาย")
        input_data = render_train_input_form()

        if st.button("ทำนายการล่าช้า"):
            prediction = predict_single_input(model, label_encoders, input_data, model.feature_names_in_)
            st.success(f"🕒 ผลการทำนาย: รถไฟจะล่าช้าประมาณ {prediction:.2f} นาที")

    except Exception as e:
        st.error(f"❌ เกิดข้อผิดพลาดในการทำนาย: {e}")
        print(e)

def render_about_page():
    st.header("ℹ️ เกี่ยวกับระบบทำนายการล่าช้าของรถไฟ")
    # Add your about page content here...

if __name__ == "__main__":
    main()