import streamlit as st
import pickle
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, cross_val_score
import pandas as pd
from .feature_processor import preprocess_features
from .model_evaluator import evaluate_model_performance

def train_model(df, features, target):
    st.info("🚂 กำลังเตรียมโมเดล Random Forest...")

    X, label_encoders = preprocess_features(df, features)
    y = df[target]

    if pd.api.types.is_datetime64_any_dtype(y):
        y = y.dt.hour * 60 + y.dt.minute

    # Define parameter grid for GridSearchCV
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [10, 20, 30, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['auto', 'sqrt']
    }

    # Initialize base model
    base_model = RandomForestRegressor(random_state=42)

    # Perform GridSearchCV
    st.info("🔍 กำลังค้นหาพารามิเตอร์ที่ดีที่สุด...")
    grid_search = GridSearchCV(
        estimator=base_model,
        param_grid=param_grid,
        cv=5,
        n_jobs=-1,
        scoring='neg_mean_squared_error'
    )
    grid_search.fit(X, y)

    # Get best model
    model = grid_search.best_estimator_

    # Perform cross-validation
    cv_scores = cross_val_score(model, X, y, cv=5, scoring='neg_mean_squared_error')
    rmse_scores = np.sqrt(-cv_scores)

    st.success(f"✅ การฝึกโมเดลเสร็จสิ้น")
    st.info(f"📊 ค่า RMSE จาก Cross-validation: {rmse_scores.mean():.2f} ± {rmse_scores.std():.2f} นาที")

    # Display feature importance
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)

    st.write("🎯 ความสำคัญของแต่ละ Feature:")
    st.dataframe(feature_importance)

    # Save model and encoders
    with open('train_model.pkl', 'wb') as f:
        pickle.dump(model, f)

    with open('label_encoders.pkl', 'wb') as f:
        pickle.dump(label_encoders, f)

    st.success("✅ บันทึกโมเดลเรียบร้อยแล้ว")

    # Evaluate model performance
    evaluate_model_performance(model, X, y)

    return model, label_encoders, X.columns
