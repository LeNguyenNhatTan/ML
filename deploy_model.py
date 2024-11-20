import joblib
from scipy import stats
import streamlit as st
import pandas as pd

st.write("""
# Dự đoán chất lượng không khí
         
Ứng dụng dùng để dự đoán **chất lượng không khí** dựa trên các yếu tố về nồng độ các chất như:
    CO, NO2, Ozone, PM2.5
"""
        )
class_names = ['Good', 'Hazardous', 'Moderate', 'Unhealthy','Unhealthy for Sensitive Groups', 'Very Unhealthy']
def input_features():
    co = st.sidebar.slider('CO', 0, 500,1)
    no2 = st.sidebar.slider('NO2', 0, 500,2)
    ozone = st.sidebar.slider('Ozone', 0, 500,3)
    pm25 = st.sidebar.slider('PM2.5', 0, 500,4)
    data = {
        'co_aqi_value' : co,
        'no2_aqi_value' : no2,
        'ozone_aqi_value' : ozone,
        'pm2.5_aqi_value' : pm25
    }
    features = pd.DataFrame(data , index=[0])
    return features
# Hàm xử lý dữ liệu trước khi dự đoán (Box-Cox Transformation)
def preprocess_input_data(data):
    # Tải giá trị lambda đã huấn luyện trước cho Box-Cox
    lambda_values = joblib.load('../pickle/boxcox_lambda_values.pkl')

    # Thêm 1 vào dữ liệu để đảm bảo nó dương (nếu cần thiết)
    data_transformed = data.copy()
    for column in data.columns:
        if (data[column] <= 0).any():  # Kiểm tra nếu có giá trị <= 0
            data_transformed[column] += 1  # Thêm 1 chỉ vào các giá trị <= 0

    # Áp dụng Box-Cox transformation cho từng cột
    for i, column in enumerate(data.columns):
        data[column]= stats.boxcox(data_transformed[column], lmbda=lambda_values[i])

    # Chuẩn hóa MinMax
    minmax_scaler = joblib.load('../pickle/minmax_scaler.pkl')
    data_scaled = minmax_scaler.transform(data)

    return data_scaled
softmax_model = joblib.load('../pickle/softmax_model.pkl')

data = input_features()
st.subheader("Dữ liệu đầu vào")
st.write(data)

data = preprocess_input_data(data)
# st.write(data)

y_pred = softmax_model.predict(data)

predicted_classes_name = class_names[y_pred[0]]

st.write('Chất lượng không khí:',predicted_classes_name)
