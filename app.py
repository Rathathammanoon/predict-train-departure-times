import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from datetime import datetime, timedelta
import pickle
import os
import warnings

warnings.filterwarnings('ignore')

plt.rcParams['font.family'] = 'Tahoma'

def load_and_preprocess_data(file_path):
    try:
        xls = pd.ExcelFile(file_path)
        st.success(f"📝 ชื่อชีทในไฟล์: {', '.join(xls.sheet_names)}")
        df = pd.read_excel(xls, 'report', engine='openpyxl')
        st.success(f"✅ โหลดข้อมูลสำเร็จ: {df.shape[0]} แถว และ {df.shape[1]} คอลัมน์")

        time_columns = ['Scheduled_departure_time_origin', 'Scheduled_arrival_time_destination',
                        'Actual_departure_time_origin', 'Actual_arrival_time_destination']
        for col in time_columns:
            if col in df.columns:
                if df[col].dtype == 'object':
                    try:
                        df[col] = pd.to_datetime(df[col], errors='coerce')
                    except:
                        st.warning(f"⚠️ ไม่สามารถแปลงคอลัมน์ {col} เป็นรูปแบบเวลาได้")
        return df
    except Exception as e:
        st.error(f"❌ เกิดข้อผิดพลาดในการโหลดข้อมูล: {e}")
        return None

def preprocess_features(df, features):
    X = df[features].copy()
    categorical_cols = X.select_dtypes(include=['object', 'category']).columns
    label_encoders = {}

    for col in categorical_cols:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
        label_encoders[col] = le

    datetime_cols = X.select_dtypes(include=['datetime64']).columns
    for col in datetime_cols:
        X[f"{col}_hour"] = X[col].dt.hour
        X[f"{col}_minute"] = X[col].dt.minute
        X.drop(col, axis=1, inplace=True)

    return X, label_encoders

def train_model(df, features, target):
    st.info("🚂 กำลังเตรียมโมเดล Random Forest...")

    X, label_encoders = preprocess_features(df, features)
    y = df[target]

    if pd.api.types.is_datetime64_any_dtype(y):
        y = y.dt.hour * 60 + y.dt.minute

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)

    st.success("✅ การฝึกโมเดลเสร็จสิ้น")

    with open('train_model.pkl', 'wb') as f:
        pickle.dump(model, f)

    with open('label_encoders.pkl', 'wb') as f:
        pickle.dump(label_encoders, f)

    st.success("✅ บันทึกโมเดลเรียบร้อยแล้ว")

    return model, label_encoders, X.columns

def predict_single_input(model, label_encoders, input_features, feature_columns):
    input_df = pd.DataFrame([input_features])

    if 'Date' in input_df.columns:
        input_df['Date'] = pd.to_datetime(input_df['Date'])

    time_columns = ['Scheduled_departure_time_origin', 'Scheduled_arrival_time_destination']
    for col in time_columns:
        if col in input_df.columns:
            if isinstance(input_df[col][0], str):
                current_date = input_df["Date"][0] if "Date" in input_df.columns else datetime.now().date()
                combined_datetime = datetime.combine(current_date, input_df[col][0])
                input_df[col] = pd.to_datetime([combined_datetime])
            else :
                input_df[col] = pd.to_datetime(input_df[col])

    for col, le in label_encoders.items():
        if col in input_df.columns:
            try:
                input_df[col] = le.transform(input_df[col].astype(str))
            except:
                st.warning(f"⚠️ ค่า {input_df[col][0]} ในคอลัมน์ {col} ไม่อยู่ในชุดข้อมูลฝึก จะใช้ค่าเริ่มต้นแทน")
                input_df[col] = 0

    datetime_cols = input_df.select_dtypes(include=['datetime64']).columns
    for col in datetime_cols:
        input_df[f"{col}_hour"] = input_df[col].dt.hour
        input_df[f"{col}_minute"] = input_df[col].dt.minute
        input_df.drop(col, axis=1, inplace=True)

    missing_cols = set(feature_columns) - set(input_df.columns)

    for col in missing_cols:
        input_df[col] = 0

    input_df = input_df[feature_columns]

    prediction = model.predict(input_df)[0]

    return prediction

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
        st.header("🧠 ฝึกโมเดลทำนายการล่าช้าของรถไฟ")

        uploaded_file = st.file_uploader("อัปโหลดไฟล์ Excel (.xlsx)", type=["xlsx"])

        if uploaded_file is not None:
            df = load_and_preprocess_data(uploaded_file)

            if df is not None:
                st.subheader("ข้อมูลตัวอย่าง")
                st.dataframe(df.head())

                st.subheader("เลือก Features และ Target")

                default_features = ['Train_type', 'Maximum_train_speed', 'Number_of_junctions',
                                   'Outbound_trips_Return_trips', 'Train_number',
                                   'Scheduled_departure_time_origin', 'Scheduled_arrival_time_destination',
                                   'Railway_line', 'Date', 'Distance', 'Number_of_stations', 'Time_period']

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

                    st.subheader("ความสำคัญของ Features")
                    feature_imp = pd.DataFrame(sorted(zip(model.feature_importances_, feature_cols)),
                                               columns=['Value', 'Feature'])
                    feature_imp = feature_imp.sort_values(by="Value", ascending=False)

                    fig, ax = plt.subplots(figsize=(10, 6))
                    sns.barplot(x="Value", y="Feature", data=feature_imp.head(10), ax=ax)
                    ax.set_title("Top 10 Feature Importance")
                    st.pyplot(fig)

                    with open('train_model.pkl', 'rb') as f:
                        model_data = f.read()

                    st.download_button(
                        label="⬇️ ดาวน์โหลดโมเดล",
                        data=model_data,
                        file_name="train_model.pkl",
                        mime="application/octet-stream"
                    )

                    with open('label_encoders.pkl', 'rb') as f:
                        encoders_data = f.read()

                    st.download_button(
                        label="⬇️ ดาวน์โหลด Label Encoders",
                        data=encoders_data,
                        file_name="label_encoders.pkl",
                        mime="application/octet-stream"
                    )

    elif page == "ทำนายการล่าช้า":
        st.header("🔍 ทำนายการล่าช้าของรถไฟ")

        model_exists = os.path.exists('train_model.pkl') and os.path.exists('label_encoders.pkl')
        if not model_exists:
            st.warning("⚠️ ยังไม่มีโมเดลที่บันทึกไว้ กรุณาอัปโหลดโมเดลหรือไปที่หน้า 'ฝึกโมเดล' ก่อน")
            model_file = st.file_uploader("อัปโหลดไฟล์โมเดล (.pkl)", type=["pkl"])
            encoders_file = st.file_uploader("อัปโหลดไฟล์ Label Encoders (.pkl)", type=["pkl"])
            if model_file and encoders_file:
                with open('train_model.pkl', 'wb') as f:
                    f.write(model_file.getbuffer())
                with open('label_encoders.pkl', 'wb') as f:
                    f.write(encoders_file.getbuffer())
                st.success("✅ อัปโหลดไฟล์สำเร็จ")
                model_exists = True
                st.experimental_rerun()

        if model_exists:
            try:
                with open('train_model.pkl', 'rb') as f:
                    model = pickle.load(f)
                with open('label_encoders.pkl', 'rb') as f:
                    label_encoders = pickle.load(f)

                st.subheader("กรอกข้อมูลเพื่อทำนาย")

                col1, col2 = st.columns(2)

                # Column ซ้าย
                with col1:
                    # ประเภทของรถไฟ
                    train_types_dict = {
                        "รถด่วนพิเศษ": 1,
                        "รถด่วน": 2,
                        "รถเร็ว": 3,
                        "รถธรรมดา": 4,
                        "ชานเมือง": 5,
                        "ท้องถิ่น": 6,
                        "พิเศษชานเมือง": 7,
                        "พิเศษโดยสาร": 8,
                        "นำเที่ยว": 9,
                        "สินค้า": 0,
                        "รถเปล่า": 11,
                        "พิเศษนำเที่ยว": 12,
                        "พิเศษ": 13,
                        "พิเศษช่วยอันตราย": 14,
                    }
                    train_type_display = st.selectbox("ประเภทรถไฟ", list(train_types_dict.keys()))
                    train_type = train_types_dict[train_type_display]

                    # ความเร็วสูงสุด
                    max_speed = st.number_input(
                        "ความเร็วสูงสุดของรถไฟ (กม./ชม.)",
                        min_value=60, max_value=160, value=120
                    )

                    # จำนวนทางแยก
                    num_junctions = st.number_input(
                        "จำนวนทางแยก",
                        min_value=0, max_value=20, value=5
                    )

                    # ขาเดินทาง
                    outbound_return_dict = {
                        "ขาไป": 1,
                        "ขากลับ": 2,
                    }
                    outbound_return_display = st.selectbox("ขาเดินทาง", list(outbound_return_dict.keys()))
                    outbound_return = outbound_return_dict[outbound_return_display]

                    # หมายเลขขบวนรถไฟ
                    train_number_list = {
                        "7": 7,
                        "8": 8
                    }
                    train_number_display = st.selectbox("หมายเลขขบวนรถไฟ", list(train_number_list.keys()))
                    train_number = train_number_list[train_number_display]

                # Column ขวา
                with col2:
                    # เวลาเดินทาง
                    travel_date = st.date_input(
                        "วันที่เดินทาง",
                    )

                    # เวลาออกเดินทางตามกำหนด
                    departure_time = st.time_input(
                        "เวลาออกเดินทางตามกำหนด",

                    )

                    if "arrival_time" not in st.session_state:
                        st.session_state.arrival_time = (datetime.now() + timedelta(minutes=30)).time()

                    # เวลาถึงจุดหมายตามกำหนด
                    arrival_time = st.time_input(
                        "เวลาถึงจุดหมายตามกำหนด",
                        value=st.session_state.arrival_time,
                        key="arrival_time"
                    )

                    # สายเดินทาง
                    railway_lines_dict = {
                        "สายเหนือ": 1,
                        "สายตะวันออกเฉียงเหนือ": 2,
                        "สายตะวันออก": 3,
                        "สายใต้": 4,
                        "วงเวียนใหญ่ - มหาชัย": 5,
                        "บ้านแหลม - แม่กลอง": 6,
                        "เชื่อมต่อสายสีแดง": 7,
                        "สายใต้ - สินค้า": 8,
                        "ข้ามภูมิภาค": 9
                    }
                    railway_type_display = st.selectbox("เส้นทางรถไฟ", list(railway_lines_dict.keys()))
                    railway_line = railway_lines_dict[railway_type_display]

                    # ระยะทาง
                    distance = st.number_input(
                        "ระยะทาง (กม.)",
                        min_value=0, max_value=1000, value=200
                    )

                    # ผ่านทั้งหมดกี่สถานี
                    num_stations = st.number_input(
                        "จำนวนสถานีที่ผ่าน",
                        min_value=0, max_value=50, value=10
                    )

                    # ช่วงเวลา
                    time_periods = {
                        "เช้า": 1,
                        "กลางวัน": 2,
                        "เย็น": 3,
                        "กลางคืน": 4,
                    }
                    time_period_display = st.selectbox("ช่วงเวลา", list(time_periods.keys()))
                    time_period = time_periods[time_period_display]

                if st.button("ทำนายการล่าช้า"):
                    departure_datetime = datetime.combine(travel_date, departure_time)
                    arrival_datetime = datetime.combine(travel_date, arrival_time)

                    input_data = {
                        'Train_type': train_type,
                        'Maximum_train_speed': max_speed,
                        'Number_of_junctions': num_junctions,
                        'Outbound_trips_Return_trips': outbound_return,
                        'Train_number': train_number,
                        'Date': travel_date,
                        'Scheduled_departure_time_origin': departure_datetime,
                        'Scheduled_arrival_time_destination': arrival_datetime,
                        'Railway_line': railway_line,
                        'Distance': distance,
                        'Number_of_stations': num_stations,
                        'Time_period': time_period
                    }

                    prediction = predict_single_input(model, label_encoders, input_data, model.feature_names_in_)
                    st.success(f"🕒 ผลการทำนาย: รถไฟจะล่าช้าประมาณ {prediction:.2f} นาที")

            except Exception as e:
                st.error(f"❌ เกิดข้อผิดพลาดในการทำนาย: {e}")
                print({e})


    elif page == "เกี่ยวกับระบบ":
        st.header("ℹ️ เกี่ยวกับระบบทำนายการล่าช้าของรถไฟ")
        st.markdown("""
        ## วัตถุประสงค์
        ระบบนี้พัฒนาขึ้นเพื่อช่วยทำนายการล่าช้าของรถไฟในประเทศไทย โดยใช้เทคนิค Machine Learning ประเภท Random Forest

        ## วิธีการใช้งาน
        1. **ฝึกโมเดล**: อัปโหลดไฟล์ Excel ที่มีข้อมูลการเดินรถไฟ เลือก Features และ Target แล้วกดปุ่ม "เริ่มฝึกโมเดล"
        2. **ทำนายการล่าช้า**: กรอกข้อมูลของรถไฟที่ต้องการทำนาย แล้วกดปุ่ม "ทำนายการล่าช้า"

        ## ข้อมูลที่ใช้ในการทำนาย
        - ประเภทรถไฟ
        - ความเร็วสูงสุดของรถไฟ
        - จำนวนทางแยก
        - ขาเดินทาง (ไป/กลับ)
        - หมายเลขขบวนรถไฟ
        - วันที่เดินทาง
        - เวลาออกเดินทางตามกำหนด
        - เวลาถึงจุดหมายตามกำหนด
        - เส้นทางรถไฟ
        - ระยะทาง
        - จำนวนสถานีที่ผ่าน
        - ช่วงเวลา

        ## เทคโนโลยีที่ใช้
        - Python 3.8+
        - Streamlit
        - Scikit-learn (Random Forest)
        - Pandas
        - Matplotlib & Seaborn

        ## ผู้พัฒนา
        พัฒนาโดย [คนหล่อเท่แบบผมนี้แหละ]
        """)

        # แสดงรูปภาพหรือแผนภาพประกอบ
        st.image("https://via.placeholder.com/800x400.png?text=Thai+Railway+System", caption="ระบบรถไฟไทย")

if __name__ == "__main__":
    main()