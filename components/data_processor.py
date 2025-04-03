import pandas as pd
import streamlit as st

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
