import pandas as pd
import streamlit as st
from datetime import datetime

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