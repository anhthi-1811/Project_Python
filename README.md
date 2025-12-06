# Hướng dẫn cài đặt và chạy

Dự án triển khai quy trình học máy hoàn chỉnh gồm 3 giai đoạn: **Tiền xử lý dữ liệu**, **Xây dựng mô hình**, và **Trực quan hóa kết quả**.

## Cấu trúc Dự Án

| File | Loại | Mô tả chức năng |
| :--- | :--- | :--- |
| **`Project_Pre.ipynb`** | Notebook | **Phần 1: Tiền xử lý.** Làm sạch, chuẩn hóa dữ liệu và Feature Engineering. |
| **`Project_Model.py`** | Script | **Phần 2: Mô hình.** Định nghĩa Class Model, cấu trúc mạng/thuật toán. |
| **`visualization.py`** | Script | **Phần 3: Trực quan hóa.** Các hàm vẽ biểu đồ, đánh giá metrics. |
| **`main.py`** | Script | **File chạy chính.** Gọi class từ `Project_Model`, thực hiện huấn luyện và hiển thị kết quả. |

## Cài đặt Môi trường (Visual Studio Code)

1. **Yêu cầu:**  
   - Python 3.x 
   - VS Code Extensions: `Python`, `Jupyter` (để chạy file .ipynb).

2. **Cài đặt thư viện:**
   Mở Terminal trong VS Code (`Ctrl` + `J`) và chạy lệnh:
   ```bash
   pip install pandas numpy scikit-learn matplotlib seaborn
   # (Thêm các thư viện khác nếu dự án bạn dùng thêm) 

## Hướng dẫn Chạy (Workflow)

Để đảm bảo chương trình hoạt động chính xác, vui lòng thực hiện tuần tự theo 2 giai đoạn sau trên Visual Studio Code:

### Giai đoạn 1: Tiền xử lý dữ liệu (Preprocessing)
**File thực hiện:** `Project_Pre.ipynb`

Bước này giúp làm sạch, chuẩn hóa và chuyển đổi dữ liệu thô trước khi đưa vào mô hình.

1. Đảm bảo file dataset gốc (input) đã nằm trong thư mục dự án.
2. Mở file **`Project_Pre.ipynb`**.
3. Tại góc trên bên phải VS Code, chọn **Select Kernel** -> **Python Environments** -> Chọn phiên bản Python bạn đang dùng.
4. Nhấn nút **Run All** trên thanh công cụ của Notebook.
   > **Kết quả:** Dữ liệu sẽ được xử lý và lưu lại (thường dưới dạng file `.csv` mới hoặc biến môi trường) để sẵn sàng cho Giai đoạn 2.

### Giai đoạn 2: Huấn luyện và Trực quan hóa
**File thực hiện:** `main.py` (File chạy chính)

File này đóng vai trò trung tâm: gọi Class mô hình từ `Project_Model.py` để training và sử dụng `visualization.py` để hiển thị kết quả.

1. Mở **Terminal** trong VS Code (Phím tắt: `Ctrl + J` hoặc `` Ctrl + ` ``).
2. Nhập lệnh sau và nhấn Enter:

   ```bash
   python main.py
