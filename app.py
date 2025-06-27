import streamlit as st
import matplotlib.pyplot as plt
import warnings
import pickle
import os
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, cross_val_score, train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder
from datetime import datetime, timedelta

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

def evaluate_model_performance(model, X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    ax1.scatter(y_test, y_pred, alpha=0.5)
    ax1.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    ax1.set_xlabel('ค่าจริง (นาที)')
    ax1.set_ylabel('ค่าทำนาย (นาที)')
    ax1.set_title('เปรียบเทียบค่าจริงกับค่าทำนาย')

    errors = y_test - y_pred
    ax2.hist(errors, bins=30)
    ax2.set_xlabel('Error (นาที)')
    ax2.set_ylabel('Frequency')
    ax2.set_title('การกระจายตัวของความคลาดเคลื่อน')

    st.pyplot(fig)

    display_metrics(mae, rmse, r2, errors)

def display_metrics(mae, rmse, r2, errors):
    with st.expander("รายละเอียดเมทริกซ์การประเมินผล"):
        col1, col2, col3 = st.columns(3)

        col1.metric(
            "MAE (Mean Absolute Error)",
            f"{mae:.2f} นาที",
            help="ค่าเฉลี่ยของความคลาดเคลื่อนสัมบูรณ์ (ยิ่งน้อยยิ่งดี)"
        )

        col2.metric(
            "RMSE (Root Mean Squared Error)",
            f"{rmse:.2f} นาที",
            help="รากที่สองของค่าเฉลี่ยของความคลาดเคลื่อนกำลังสอง (ยิ่งน้อยยิ่งดี)"
        )

        col3.metric(
            "R² Score",
            f"{r2:.4f}",
            help="สัมประสิทธิ์การตัดสินใจ (ยิ่งใกล้ 1 ยิ่งดี)"
        )

        st.write("สถิติของความคลาดเคลื่อน:")
        st.write(f"- ค่าเฉลี่ยของความคลาดเคลื่อน: {errors.mean():.2f} นาที")
        st.write(f"- ส่วนเบี่ยงเบนมาตรฐานของความคลาดเคลื่อน: {errors.std():.2f} นาที")
        st.write(f"- ความคลาดเคลื่อนต่ำสุด: {errors.min():.2f} นาที")
        st.write(f"- ความคลาดเคลื่อนสูงสุด: {errors.max():.2f} นาที")

def train_model(df, features, target):
    st.info("🚂 กำลังเตรียมโมเดล Random Forest...")

    X, label_encoders = preprocess_features(df, features)
    y = df[target]

    if pd.api.types.is_datetime64_any_dtype(y):
        y = y.dt.hour * 60 + y.dt.minute

    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [10, 20, 30, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['auto', 'sqrt']
    }

    base_model = RandomForestRegressor(random_state=42)

    st.info("🔍 กำลังค้นหาพารามิเตอร์ที่ดีที่สุด...")
    grid_search = GridSearchCV(
        estimator=base_model,
        param_grid=param_grid,
        cv=5,
        n_jobs=-1,
        scoring='neg_mean_squared_error'
    )
    grid_search.fit(X, y)

    model = grid_search.best_estimator_

    cv_scores = cross_val_score(model, X, y, cv=5, scoring='neg_mean_squared_error')
    rmse_scores = np.sqrt(-cv_scores)

    st.success(f"✅ การฝึกโมเดลเสร็จสิ้น")
    st.info(f"📊 ค่า RMSE จาก Cross-validation: {rmse_scores.mean():.2f} ± {rmse_scores.std():.2f} นาที")

    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)

    st.write("🎯 ความสำคัญของแต่ละ Feature:")
    st.dataframe(feature_importance)

    with open('train_model.pkl', 'wb') as f:
        pickle.dump(model, f)

    with open('label_encoders.pkl', 'wb') as f:
        pickle.dump(label_encoders, f)

    st.success("✅ บันทึกโมเดลเรียบร้อยแล้ว")

    evaluate_model_performance(model, X, y)

    return model, label_encoders, X.columns

def predict_single_input(model, label_encoders, input_features, feature_columns):
    input_df = pd.DataFrame([input_features])

    if 'Day_Month_Year' in input_df.columns:
        travel_date = input_df['Day_Month_Year'][0]

        if 'Date' in feature_columns and 'Date' in input_df.columns:
            pass
        elif 'Date' in feature_columns and 'Date' not in input_df.columns:
            input_df['Date'] = travel_date.isoweekday()

    time_columns = ['Scheduled_departure_time_origin', 'Scheduled_arrival_time_destination']
    for col in time_columns:
        if col in input_df.columns:
            if isinstance(input_df[col][0], str):
                current_date = input_df["Day_Month_Year"][0] if "Day_Month_Year" in input_df.columns else datetime.now().date()
                combined_datetime = datetime.combine(current_date, datetime.strptime(input_df[col][0], "%H:%M:%S").time())
                input_df[col] = pd.to_datetime([combined_datetime])
            elif isinstance(input_df[col][0], datetime):
                pass
            else:
                current_date = input_df["Day_Month_Year"][0] if "Day_Month_Year" in input_df.columns else datetime.now().date()
                combined_datetime = datetime.combine(current_date, input_df[col][0])
                input_df[col] = pd.to_datetime([combined_datetime])

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

def render_train_input_form():
    col1, col2 = st.columns(2)

    with col1:
        train_type = render_train_type_input()
        max_speed = render_max_speed_input()
        num_junction = render_junction_input()
        outbound_return = render_direction_input()
        train_number = render_train_number_input()
        day_of_week = render_day_of_week_input()
        departure_delay = st.number_input("ความล่าช้าจากต้นทาง", min_value=0, max_value=2000, step=1)
        actual_departure_time = st.time_input("เวลาจริงที่ต้นทางออกตัว")
        actual_arrival_time = st.time_input("เวลาจริงที่ถึงปลายทาง")

    with col2:
        departure_time = st.time_input("เวลาออกเดินทางตามกำหนด")

        if "arrival_time" not in st.session_state:
            st.session_state.arrival_time = (datetime.now() + timedelta(minutes=30)).time()

        arrival_time = st.time_input(
            "เวลาถึงจุดหมายตามกำหนด",
            value=st.session_state.arrival_time,
            key="arrival_time"
        )

        railway_line = render_railway_line_input()
        distance = st.number_input("ระยะทาง (กม.)", min_value=31, max_value=1152, value=31)
        num_stations = st.number_input("จำนวนสถานีที่ผ่าน", min_value=2, max_value=93, value=2)
        time_period = render_time_period_input()
        max_delay = st.number_input("ความล่าช้าสูงสุด", min_value=0, max_value=5000, step=1)
        total_delay = st.number_input("รวมเวลาทั้งหมดที่ล่าช้า", min_value=0, max_value=15000, step=1)
        stations_delay = st.number_input("จำนวนสถานีทั้งหมดที่ล่าช้า", min_value=0, max_value=93, step=1)

    return {
        'Train_type': train_type,
        'Maximum_train_speed': max_speed,
        'Number_of_junctions': num_junction,
        'Outbound_trips_Return_trips': outbound_return,
        'Train_number': train_number,
        'Date': day_of_week,
        'Departure_delay_origin': departure_delay,
        'Actual_departure_time_origin_hour': actual_departure_time.hour,
        'Actual_departure_time_origin_minute': actual_departure_time.minute,
        'Actual_arrival_time_destination_hour': actual_arrival_time.hour,
        'Actual_arrival_time_destination_minute': actual_arrival_time.minute,
        'Scheduled_departure_time_origin_hour': departure_time.hour,
        'Scheduled_departure_time_origin_minute': departure_time.minute,
        'Scheduled_arrival_time_destination_hour': arrival_time.hour,
        'Scheduled_arrival_time_destination_minute': arrival_time.minute,
        'Railway_line': railway_line,
        'Distance': distance,
        'Number_of_stations': num_stations,
        'Time_period': time_period,
        'Maximum_delay': max_delay,
        'Total_delay_time': total_delay,
        'Number_of_delayed_stations': stations_delay
    }

def render_train_type_input():
    train_types_dict = {
        "รถด่วนพิเศษ": 1, "รถด่วน": 2, "รถเร็ว": 3,
        "รถธรรมดา": 4, "ชานเมือง": 5, "ท้องถิ่น": 6,
        "พิเศษชานเมือง": 7, "พิเศษโดยสาร": 8, "นำเที่ยว": 9,
        "สินค้า": 0, "รถเปล่า": 11, "พิเศษนำเที่ยว": 12,
        "พิเศษ": 13, "พิเศษช่วยอันตราย": 14,
    }
    train_type_display = st.selectbox("ประเภทรถไฟ", list(train_types_dict.keys()))
    return train_types_dict[train_type_display]

def render_max_speed_input():
    max_speed_types_dict = {
        "60": 60, "100": 100, "120": 120, "160": 160,
    }
    max_speed_display = st.selectbox("ความเร็วสูงสุด (กม./ชม.)", list(max_speed_types_dict.keys()))
    return max_speed_types_dict[max_speed_display]

def render_junction_input():
    num_junction_types_dict = {
          "0": 0, "1": 1, "2": 2, "3": 3, "4": 4, "5": 5,
    }
    num_junction_display = st.selectbox("จำนวนทางแยก", list(num_junction_types_dict.keys()))
    return num_junction_types_dict[num_junction_display]

def render_direction_input():
    outbound_return_types_dict = {
        "ขาไป": 1, "ขากลับ": 2,
    }
    outbound_return_display = st.selectbox("ทางออก-กลับ", list(outbound_return_types_dict.keys()))
    return outbound_return_types_dict[outbound_return_display]

def render_train_number_input():
    train_number_types_dict = {
    '7': 7, '8': 8, '9': 9, '10': 10, '13': 13, '14': 14, '21': 21, '22': 22,
    '23': 23, '25': 25, '26': 26, '31': 31, '32': 32, '37': 37, '38': 38,
    '39': 39, '40': 40, '43': 43, '45': 45, '46': 46, '51': 51, '71': 71,
    '75': 75, '76': 76, '83': 83, '84': 84, '85': 85,
    '86': 86, '102': 102, '107': 107, '109': 109, '111': 111,
    '112': 112, '133': 133, '134': 134, '135': 135, '136': 136,
    '139': 139, '140': 140, '141': 141, '142': 142, '147': 147,
    '148': 148, '167': 167, '168': 168, '169': 169, '170': 170,
    '171': 171, '172': 172, '201': 201, '202': 202, '207': 207,
    '208': 208, '209': 209, '210': 210, '211': 211, '212': 212,
    '233': 233, '234': 234, '251': 251, '252': 252, '254': 254,
    '255': 255, '257': 257, '258': 258, '259': 259, '260': 260,
    '261': 261, '262': 262, '275': 275, '276': 276, '277': 277,
    '278': 278, '279': 279, '280': 280, '281': 281, '282': 282,
    '283': 283, '284': 284, '301': 301, '302': 302, '303': 303,
    '304': 304, '313': 313, '314': 314, '317': 317, '318': 318,
    '339': 339, '340': 340, '341': 341, '342': 342, '351': 351,
    '352': 352, '355': 355, '356': 356, '367': 367, '368': 368,
    '371': 371, '372': 372, '379': 379, '380': 380, '383': 383,
    '384': 384, '388': 388, '389': 389, '390': 390, '391': 391,
    '401': 401, '402': 402, '403': 403, '405': 405, '406': 406,
    '407': 407, '408': 408, '410': 410, '415': 415, '416': 416,
    '417': 417, '418': 418, '419': 419, '420': 420, '421': 421,
    '422': 422, '423': 423, '424': 424, '425': 425, '426': 426,
    '427': 427, '428': 428, '429': 429, '430': 430, '431': 431,
    '432': 432, '433': 433, '434': 434, '439': 439, '440': 440,
    '445': 445, '446': 446, '447': 447, '448': 448, '451': 451,
    '452': 452, '453': 453, '454': 454, '455': 455, '456': 456,
    '463': 463, '464': 464, '485': 485, '486': 486, '489': 489,
    '490': 490, '721': 721, '722': 722, '831': 831, '832': 832,
    '833': 833, '834': 834, '835': 835, '836': 836, '837': 837,
    '838': 838, '839': 839, '840': 840, '841': 841, '842': 842,
    '843': 843, '844': 844, '845': 845, '846': 846, '847': 847,
    '848': 848, '849': 849, '850': 850, '851': 851, '852': 852,
    '853': 853, '854': 854, '855': 855, '856': 856, '857': 857,
    '858': 858, '859': 859, '860': 860, '909': 909, '910': 910,
    '911': 911, '912': 912, '947': 947, '948': 948, '949': 949,
    '950': 950, '985': 985, '986': 986, '997': 997, '998': 998,
    '1115': 1115, '1116': 1116, '1117': 1117, '1118': 1118, '1123': 1123,
    '1124': 1124, '1129': 1129, '1132': 1132, '4302': 4302, '4303': 4303,
    '4304': 4304, '4305': 4305, '4306': 4306, '4307': 4307, '4308': 4308,
    '4309': 4309, '4310': 4310, '4311': 4311, '4312': 4312, '4313': 4313,
    '4314': 4314, '4315': 4315, '4316': 4316, '4317': 4317, '4320': 4320,
    '4321': 4321, '4322': 4322, '4323': 4323, '4324': 4324, '4325': 4325,
    '4326': 4326, '4327': 4327, '4328': 4328, '4329': 4329, '4340': 4340,
    '4341': 4341, '4342': 4342, '4343': 4343, '4344': 4344, '4345': 4345,
    '4346': 4346, '4347': 4347, '4380': 4380, '4381': 4381, '4382': 4382,
    '4383': 4383, '4384': 4384, '4385': 4385, '4386': 4386, '4387': 4387
    }
    train_number_display = st.selectbox("หมายเลขรถไฟ", list(train_number_types_dict.keys()))
    return train_number_types_dict[train_number_display]

def render_day_of_week_input():
    day_of_week_types_dict = {
        "จันทร์": 1, "อังคาร": 2, "พุธ": 3, "พฤหัสบดี": 4, "ศุกร์": 5, "เสาร์": 6, "อาทิตย์": 7,
    }
    day_of_week_display = st.selectbox("วันในสัปดาห์", list(day_of_week_types_dict.keys()))
    return day_of_week_types_dict[day_of_week_display]

def render_railway_line_input():
    railway_line_types_dict = {
        "สายเหนือ": 1,"สายตะวันออกเฉียงเหนือ": 2,"สายตะวันออก": 3,
        "สายใต้": 4,"วงเวียนใหญ่ - มหาชัย": 5,"บ้านแหลม - แม่กลอง": 6,
        "เชื่อมต่อสายสีแดง": 7,"สายใต้ - สินค้า": 8,"ข้ามภูมิภาค": 9
    }
    railway_line_display = st.selectbox("สายรถไฟ", list(railway_line_types_dict.keys()))
    return railway_line_types_dict[railway_line_display]

def render_time_period_input():
    time_period_types_dict = {
      "กลางวัน": 1,"กลางคืน": 2,
    }
    time_period_display = st.selectbox("ช่วงเวลา", list(time_period_types_dict.keys()))
    return time_period_types_dict[time_period_display]

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