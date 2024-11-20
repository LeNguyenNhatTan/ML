# Air Quality Classification using Softmax Regression
## 📄 Giới thiệu
Đây là một ứng dụng Streamlit được xây dựng để phân loại chất lượng không khí dựa trên dữ liệu đầu vào. Ứng dụng sử dụng Softmax Regression, một thuật toán học máy tuyến tính phù hợp để phân loại nhiều lớp.

Ứng dụng cho phép người dùng nhập thông số không khí (như PM2.5, PM10, CO, NO₂, O₃, v.v.) và dự đoán chất lượng không khí thuộc các mức như:

- **Tốt (Good)**
- **Trung bình (Moderate)**
- **Không lành mạnh (Unhealthy)**
- **Không lành mạnh đối với nhóm nhạy cảm (Unhealthy for Sensitive Groups)**
- **Rất không lành mạnh (Very Unhealthy)**
- **Nguy hiểm (Hazardous)**

## 🚀 Cách sử dụng
### Yêu cầu hệ thống
Python >= 3.7

### Hướng dẫn cài đặt
1. **Clone repository:**
git clone [https://github.com/your-repo-name/air-quality-classification.git]

2. **Cài đặt thư viện:**
pip install -r requirements.txt  

3. **Chạy ứng dụng Streamlit:**
streamlit run app.py  

4. **Truy cập ứng dụng:**
Sau khi chạy lệnh trên, ứng dụng sẽ được triển khai tại http://localhost:8501/.

5. **Sử dụng giao diện:**
Nhập các thông số không khí (PM2.5, PM10, CO, NO₂, O₃, v.v.) vào biểu mẫu.
Nhấn nút Dự đoán để xem kết quả phân loại.

## ⚙️ Mô hình Softmax Regression
Cách hoạt động: Softmax Regression sử dụng hàm Softmax để tính xác suất cho từng lớp dự đoán. Lớp có xác suất cao nhất sẽ được chọn làm kết quả.
Huấn luyện: Mô hình được huấn luyện trên tập dữ liệu chất lượng không khí với các tính năng liên quan đến ô nhiễm không khí và điều kiện môi trường.
Các bước triển khai:
Tiền xử lý dữ liệu:

Loại bỏ dữ liệu bị thiếu hoặc không hợp lệ.
Chuẩn hóa dữ liệu để cải thiện hiệu suất mô hình.
Huấn luyện mô hình:

Sử dụng thư viện scikit-learn để xây dựng và huấn luyện Softmax Regression.
Chia tập dữ liệu thành train và test để đánh giá mô hình.
Triển khai ứng dụng:

Tích hợp mô hình đã huấn luyện vào Streamlit để tạo giao diện người dùng.
## 📊 Dữ liệu mẫu
Tập dữ liệu sử dụng cho dự án này bao gồm các thông số về chất lượng không khí, chẳng hạn:

PM2.5, PM10, CO, NO₂, O₃.
Dữ liệu được lấy từ các nguồn công khai như Kaggle hoặc OpenAQ.

## ✨ Tính năng nổi bật
Trực quan hóa dữ liệu: Hiển thị dữ liệu đầu vào và kết quả phân loại thông qua biểu đồ.
Dự đoán nhanh: Nhận kết quả phân loại chỉ trong vài giây.
Thân thiện với người dùng: Giao diện dễ sử dụng, phù hợp cho cả người dùng không chuyên về kỹ thuật.

## 💡 Phát triển thêm
Tích hợp thêm các thông số môi trường khác như nhiệt độ, độ ẩm.
Nâng cấp mô hình phân loại bằng các thuật toán phức tạp hơn như Random Forest, Neural Networks.
Xây dựng API để tích hợp vào các ứng dụng di động hoặc web khác.
## 🛠️ Đóng góp
Nếu bạn muốn đóng góp cho dự án, hãy tạo Pull Request hoặc mở Issue trên GitHub.

## 📧 Liên hệ
Nếu có bất kỳ câu hỏi hoặc góp ý nào, vui lòng liên hệ qua email: nhattan425@gmail.com.

Chúc bạn có trải nghiệm thú vị với ứng dụng! 🎉
