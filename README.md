# Car Price Prediction Project

Dự án xây dựng quy trình học máy để dự đoán giá xe hơi. Hệ thống tự động hóa hoàn toàn từ khâu xử lý dữ liệu thô, huấn luyện đa mô hình, đến giải thích kết quả bằng SHAP.

## Cấu trúc Dự Án

```text
.
├── main.py                    # File chạy chính (Master Script)
├── Data/
│   ├── Raw/
│   │   └── Car details v3.csv # Dữ liệu đầu vào thô
│   ├── Processed/             # Chứa dữ liệu sạch sau khi xử lý
│   │   ├── data_processed.csv
│   │   └── data_encoders.csv
│   └── EDA/                   # Dữ liệu phục vụ phân tích
│       └── Car_details_eda.csv
└── Source/                    # Source code & Logic
    ├── Data_processing.ipynb  # Notebook quy trình xử lý
    ├── preprocessor.py        # Class DataPreprocessor
    ├── Project_Model.py       # Class ModelTrainer
    ├── visualization.py       # Class Visualizer
    └── README.md              # File hướng dẫn này
---

## Quy trình Hoạt động (Automated Pipeline)

Khi chạy `main.py`, hệ thống sẽ thực hiện tuần tự 4 giai đoạn:

1.  **Tiền xử lý (Preprocessing):**
    * Tự động gọi logic từ `DataPreprocessor` (trong Notebook).
    * Làm sạch, chuẩn hóa, mã hóa và xuất ra file `data_processed.csv`.
2.  **Huấn luyện (Modeling):**
    * Class `ModelTrainer` tách tập dữ liệu Train/Test.
    * Huấn luyện hàng loạt mô hình và tìm ra thuật toán tối ưu nhất.
3.  **Đánh giá (Evaluation):**
    * Class `Visualizer` vẽ biểu đồ phân phối và so sánh hiệu năng các model.
4.  **Giải thích (Explainability):**
    * Tính toán Feature Importance và vẽ biểu đồ SHAP để giải thích lý do đằng sau giá xe dự đoán.

---

## Cài đặt Môi trường

1.  **Yêu cầu:** Python 3.8+, VS Code.
2.  **Cài đặt thư viện:**
    Mở Terminal tại thư mục dự án và chạy:
    ```bash
    pip install pandas numpy scikit-learn matplotlib seaborn shap
    # Cần cài thêm jupyter hoặc ipykernel nếu main.py gọi notebook thông qua lệnh hệ thống
    pip install jupyter
    ```

---

## Hướng dẫn Chạy (Workflow)

Bạn chỉ cần thực hiện **duy nhất 1 bước** để chạy toàn bộ dự án:

1.  Mở thư mục dự án trong VS Code.
2.  Mở **Terminal** (`Ctrl + J`).
3.  Chạy lệnh:

    ```bash
    python main.py
    ```

**Quá trình tự động diễn ra:**
* **Step 1:** Hệ thống tự động xử lý dữ liệu thô (bạn sẽ thấy các file `.csv` mới xuất hiện).
* **Step 2:** Quá trình Training bắt đầu (Terminal hiển thị các chỉ số Accuracy/Loss).
* **Step 3:** Kết thúc, các biểu đồ phân tích và SHAP sẽ tự động hiển thị (popup) hoặc được lưu lại.

---

## Kết quả Đầu ra (Outputs)

Sau khi chương trình chạy xong, bạn sẽ nhận được:

* **Model:** File lưu model tốt nhất (ví dụ `.pkl`).
* **Data:** 3 file CSV đã được làm sạch và mã hóa.
* **Charts:**
    * Biểu đồ phân phối giá xe.
    * Biểu đồ so sánh hiệu năng các thuật toán.
    * Biểu đồ SHAP (tác động của từng tính năng lên giá).
