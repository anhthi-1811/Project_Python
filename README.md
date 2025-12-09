# Car Price Prediction Project

Dự án xây dựng quy trình học máy để dự đoán giá xe hơi. Hệ thống bao gồm đầy đủ các bước từ xử lý dữ liệu thô, huấn luyện đa mô hình, đánh giá hiệu năng và giải thích kết quả dự đoán bằng SHAP.

## Cấu trúc Dự Án

| File / Thư mục | Loại | Mô tả chức năng |
| :--- | :--- | :--- |
| **`Data_processing.ipynb`** | Notebook | **Bước 1:** Chạy quy trình tiền xử lý dữ liệu. |
| **`main.py`** | Script | **Bước 2:** File chạy chính để huấn luyện, đánh giá và giải thích mô hình. |
| **`src/`** (hoặc root) | Classes | Chứa các Class logic: `DataPreprocessor`, `ModelTrainer`, `Visualizer`. |
| `data_processed.csv` | Output | Dữ liệu sạch sau khi xử lý (dùng để train). |
| `Car_details_eda.csv` | Output | Dữ liệu phục vụ phân tích EDA. |
| `data_encoders.csv` | Output | File chứa thông tin mã hóa (mapping). |

---

## Quy trình Hoạt động (Pipeline)

Hệ thống hoạt động dựa trên 4 giai đoạn chính, sử dụng các Class chuyên biệt:

### 1. Tiền xử lý dữ liệu (Preprocessing)
*Sử dụng Class: `DataPreprocessor` (trong `Data_processing.ipynb`)*
- **Làm sạch:** Xử lý dữ liệu thô, missing values và outliers.
- **Biến đổi:** Chuẩn hóa (Scaling) và Mã hóa (Encoding).
- **Feature Engineering:** Tạo các cột đặc trưng mới.
- **Output:** Xuất ra 3 file CSV (`data_processed.csv`, `Car_details_eda.csv`, `data_encoders.csv`).

### 2. Huấn luyện Mô hình (Modeling)
*Sử dụng Class: `ModelTrainer` (được gọi bởi `main.py`)*
- Tách tập dữ liệu Train/Test.
- Huấn luyện song song nhiều thuật toán Machine Learning.
- So sánh và tìm ra mô hình dự đoán giá tốt nhất.
- Lưu trữ mô hình tốt nhất (Best Model).

### 3. Đánh giá Dữ liệu & Hiệu năng
*Sử dụng Class: `Visualizer` (được gọi bởi `main.py`)*
- Vẽ biểu đồ phân phối dữ liệu.
- Xem xét mối tương quan (Correlation) giữa các thuộc tính.
- Vẽ biểu đồ so sánh hiệu năng (Accuracy, MAE, RMSE...) của các mô hình.

### 4. Giải thích Mô hình (Explainability)
*Sử dụng Class: `Visualizer` (được gọi bởi `main.py`)*
- **Feature Importance:** Xác định biến nào ảnh hưởng nhất đến giá xe.
- **SHAP Analysis:** Giải thích chi tiết quan hệ giữa đặc trưng và giá trị dự đoán (tại sao xe này lại có giá đó).

---

## Cài đặt & Yêu cầu hệ thống

### 1. Môi trường
- **Python:** 3.8+
- **Editor:** Visual Studio Code (Khuyên dùng).
- **Extensions VS Code:** Python, Jupyter.

### 2. Cài đặt thư viện
Mở Terminal tại thư mục dự án và chạy lệnh:

```bash
pip install pandas numpy scikit-learn matplotlib seaborn shap
