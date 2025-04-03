import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

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
