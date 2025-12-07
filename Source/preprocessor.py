import pandas as pd
import numpy as np
import re
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder, OneHotEncoder
import matplotlib.pyplot as plt

class DataPreprocessor:
    def __init__(self, df=None):
        """
        Khởi tạo đối tượng DataPreprocessor với dataframe
        df lưu trữ dữ liệu cần xử lý
        scaler lưu trữ đối tượng scaler đã fit (nếu có)
        encoders lưu trữ các bộ mã hóa phân loại đã fit (nếu có)
        """
        self.df = df
        self.numeric_cols = []
        self.categorical_cols = []
        self.target = None
        self.scaler = None
        self.encoders = {}
    def set_target(self, target_col):
        """Đặt cột mục tiêu cho DataPreprocessor"""
        if target_col not in self.df.columns:
            raise ValueError(f"Cột {target_col} không tồn tại trong DataFrame")
        self.target = target_col
    def set_numeric_cols(self, cols):
        """Đặt cột số cho DataPreprocessor"""
        for col in cols:
            if col not in self.df.columns:
                raise ValueError(f"Cột {col} không tồn tại trong DataFrame")
        self.numeric_cols = list(cols)
    def set_categorical_cols(self, cols):
        """Đặt cột phân loại cho DataPreprocessor"""
        for col in cols:
            if col not in self.df.columns:
                raise ValueError(f"Cột {col} không tồn tại trong DataFrame")
        self.categorical_cols = list(cols)
    def __repr__(self):
        """Hiển thị thông tin về đối tượng DataPreprocessor"""
        return str(self.df.info())

    # Đọc file
    @staticmethod
    def read_file(path):
        """Đọc file dữ liệu từ đường dẫn và trả về đối tượng DataPreprocessor (chỉ có thể đọc file .csv, .xlsx, .json)"""
        try:
            if path.endswith(".csv"):
                df = pd.read_csv(path)
            elif path.endswith(".xlsx") or path.endswith(".xls"):
                df = pd.read_excel(path)
            elif path.endswith(".json"):
                df = pd.read_json(path)
            else:
                raise ValueError("Định dạng không hỗ trợ. Chỉ hỗ trợ CSV, XLSX, JSON.")
            return DataPreprocessor(df)
        except Exception as e:
            print("Lỗi đọc file:", e)
            return None

    # Xử lý hàng trùng lặp
    def remove_duplicates(self):
        """Xóa các hàng trùng lặp trong dataframe"""
        initial_count = len(self.df)
        self.df = self.df.drop_duplicates().reset_index(drop=True)
        final_count = len(self.df)
        print(f"Đã xóa {initial_count - final_count} hàng trùng lặp.")
        return self
    # Phân loại cột
    def infer_column_types(self):
        """ Xác định cột số và cột phân loại."""
        numeric_cols = []
        categorical_cols = []
        for col in self.df.columns:
            # Đếm số giá trị khác nhau
            unique_count = self.df[col].nunique()
            if unique_count <= 15:
                categorical_cols.append(col)
            elif pd.api.types.is_numeric_dtype(self.df[col]):
                numeric_cols.append(col)
        self.set_numeric_cols(numeric_cols)
        self.set_categorical_cols(categorical_cols)
        return self
    # Xử lý giá trị thiếu
    def fill_missing(self, numeric_method="median", categorical_method="mode"):
        """
        Điền giá trị thiếu cho dataframe.
        - numeric_method: 'mean', 'median', 'mode', 'ffill', 'bfill'
        - categorical_method: 'mode', 'ffill', 'bfill'
        mean: trung bình
        median: trung vị
        mode: giá trị xuất hiện nhiều nhất
        ffill: điền giá trị trước đó
        bfill: điền giá trị sau đó
        Nếu phương pháp không hợp lệ, sẽ xuất ra lỗi ValueError.
        Mặc định là median cho cột số và mode cho cột phân loại.
        """

        # Xử lý giá trị thiếu cho cột số 
        for col in self.numeric_cols:
            if numeric_method == "mean":
                self.df[col] = self.df[col].fillna(self.df[col].mean())
            elif numeric_method == "median":
                self.df[col] = self.df[col].fillna(self.df[col].median())
            elif numeric_method == "mode":
                self.df[col] = self.df[col].fillna(self.df[col].mode()[0])
            elif numeric_method == "ffill":
                self.df[col] = self.df[col].fillna(method="ffill")
            elif numeric_method == "bfill":
                self.df[col] = self.df[col].fillna(method="bfill")
            else:
                raise ValueError("numeric_method không hợp lệ")

        # Xử lý giá trị thiếu cho cột phân loại 
        for col in self.categorical_cols:
            if categorical_method == "mode":
                self.df[col] = self.df[col].fillna(self.df[col].mode()[0])
            elif categorical_method == "ffill":
                self.df[col] = self.df[col].fillna(method="ffill")
            elif categorical_method == "bfill":
                self.df[col] = self.df[col].fillna(method="bfill")
            else:
                raise ValueError("categorical_method không hợp lệ")

        return self
    # Phát hiện ngoại lai và nhận xét
    def detect_outliers(self, method="iqr", z_thresh=3, iqr_factor=1.5):
        """
        Phát hiện ngoại lai bằng Z-score, IQR hoặc Isolation Forest và nhận xét.
        Nếu phương pháp không hợp lệ, sẽ xuất ra lỗi ValueError.
        - method: 'zscore', 'iqr', 'isolation_forest'
        - z_thresh: ngưỡng Z-score (chỉ dùng khi method là 'zscore')
        - iqr_factor: hệ số IQR (chỉ dùng khi method là 'iqr')
        Mặc định phương pháp là IQR.
        """
        if len(self.numeric_cols) == 0:
            print("Không có cột số để phát hiện ngoại lai.")
            return {}
        
        report = {} # Lưu trữ báo cáo

        for col in self.numeric_cols:
            data = self.df[col].dropna()
            outliers_idx = []

            # Z-SCORE
            if method == "zscore":
                z_scores = (data - data.mean()) / data.std()
                outliers_idx = data[np.abs(z_scores) > z_thresh].index

            # IQR METHOD
            elif method == "iqr":
                Q1 = data.quantile(0.25)
                Q3 = data.quantile(0.75)
                IQR = Q3 - Q1
                lower = Q1 - iqr_factor * IQR
                upper = Q3 + iqr_factor * IQR
                outliers_idx = data[(data < lower) | (data > upper)].index

            # ISOLATION FOREST
            elif method == "isolation_forest":
                model = IsolationForest(contamination="auto", random_state=42)
                labels = model.fit_predict(data.to_frame())
                outliers_idx = data[labels == -1].index

            else:
                raise ValueError("method phải là 'zscore', 'iqr', hoặc 'isolation_forest'")

            # Lưu thống kê
            count = len(outliers_idx)
            total = len(data)
            percent = (count / total) * 100
            if count > 0:
                deviation = np.mean(np.abs(data.loc[outliers_idx] - data.mean()))
            else:
                deviation = 0

            report[col] = {
                "total": total,
                "outliers": count,
                "percent": round(percent, 2),
                "mean_difference": round(deviation, 4)
            }

        # In báo cáo nhận xét
        print("\nBÁO CÁO NGOẠI LAI")
        print(f"    Phương pháp: {method.upper()}\n")

        for col, info in report.items():
            print(f"Cột: {col}")
            print(f"   - Tổng số giá trị: {info['total']}")
            print(f"   - Số ngoại lai: {info['outliers']} ({info['percent']}%)")

            if info["outliers"] > 0:
                print(f"   - Độ lệch trung bình so với mean: {info['mean_difference']}")
                if info["percent"] > 5:
                    print("   Nhận xét: Nhiều ngoại lai — có thể ảnh hưởng mô hình.")
                elif info["percent"] > 1:
                    print("   Nhận xét: Số giá trị ngoại lai bình thường — chấp nhận được.")
                else:
                    print("   Nhận xét: Ít ngoại lai — không đáng lo ngại.")
            else:
                print("   Không phát hiện ngoại lai.")
            print()

            # Vẽ Boxplot
            plt.figure(figsize=(5, 4))
            plt.boxplot(self.df[col].dropna())
            plt.title(f"Boxplot — {col}")
            plt.ylabel(col)
            plt.grid(True, linestyle="--", alpha=0.5)
            plt.show()
        return report

    # Chuẩn hóa dữ liệu số
    def scale_numeric(self, method="standard"):
        """
        Chuẩn hóa dữ liệu số bằng:
        - method="standard": StandardScaler (Z-score normalization)
        - method="minmax": MinMaxScaler (đưa dữ liệu về [0, 1])
        Nếu phương pháp không hợp lệ, sẽ xuất ra lỗi ValueError.
        Mặc định là StandardScaler.
        """

        if len(self.numeric_cols) == 0:
            print("Không có cột số để chuẩn hóa.")
            return self
        # StandardScaler
        if method == "standard":
            self.scaler = StandardScaler()
            print("Sử dụng StandardScaler (Chuẩn hóa Z-score)")
        # MinMaxScaler
        elif method == "minmax":
            self.scaler = MinMaxScaler()
            print("Sử dụng MinMaxScaler (Scale về khoảng 0–1)")
        else:
            raise ValueError("method phải là 'standard' hoặc 'minmax'!")

        # Fit và transform
        scaled_values = self.scaler.fit_transform(self.df[self.numeric_cols])
        for i, col in enumerate(self.numeric_cols):
            self.df[col] = scaled_values[:, i]

        print(f"Đã chuẩn hóa {len(self.numeric_cols)} cột số:", self.numeric_cols)
        return self

    # Mã hóa phân loại
    def encode_categories(self, method="label", custom_mappings=None):
        """
        Mã hóa các biến phân loại.
        method: 'label' hoặc 'onehot'
            'label' => dùng LabelEncoder
            'onehot' => dùng OneHotEncoder
        custom_mappings: dict, ví dụ {'fuel': {'Petrol': 0, 'Diesel': 1, ...}}
            Sử dụng để chuyển cột dạng text sang số theo mapping tự định nghĩa.
        """
        
        # Custom mapping
        if custom_mappings:
            for col, mapping in custom_mappings.items():
                if col in self.df:
                    self.df[col] = self.df[col].map(mapping).fillna(-1)  # điền -1 cho giá trị không trong mapping
                    self.encoders[col] = mapping  # lưu mapping
            return self
        # LabelEncoder hoặc OneHotEncoder
        for col in self.categorical_cols:
            # Bỏ qua cột mục tiêu
            if col == self.target:
                continue
            # Nếu cột là kiểu số thì bỏ qua
            if pd.api.types.is_numeric_dtype(self.df[col]):
                continue
            if custom_mappings and col in custom_mappings:
                continue  # đã map rồi, bỏ qua

            if method == "label":
                le = LabelEncoder()
                self.df[col] = le.fit_transform(self.df[col].astype(str))
                self.encoders[col] = le

            elif method == "onehot":
                ohe = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
                transformed = ohe.fit_transform(self.df[[col]])
                new_cols = [f"{col}_{cat}" for cat in ohe.categories_[0]]
                self.df[new_cols] = transformed
                self.set_categorical_cols(self.categorical_cols + new_cols)
                self.encoders[col] = ohe
            else:
                raise ValueError("method phải là 'label' hoặc 'onehot'")

        return self
    # Khôi phục giá trị gốc sau khi chuẩn hóa
    def inverse_scale(self, data=None):
        """
        Khôi phục giá trị gốc trước khi chuẩn hóa (inverse transform).
        - Nếu data=None → thực hiện inverse cho toàn bộ các cột đã scale trong self.df.
        - Nếu data là DataFrame / Series → inverse transform và trả về bản khôi phục.
        """

        if self.scaler is None:
            raise ValueError("Không có scaler nào được fit. Hãy chạy scale_numeric() trước.")

        # Các cột cần inverse (loại bỏ target)
        cols_to_inverse = [c for c in self.numeric_cols if c != self.target]

        # Dữ liệu đầu vào
        if data is None:
            input_df = self.df[cols_to_inverse]
        else:
            input_df = data[cols_to_inverse]

        # Thực hiện inverse transform
        restored = self.scaler.inverse_transform(input_df)

        # Gán lại vào df nếu data=None
        if data is None:
            for i, col in enumerate(cols_to_inverse):
                self.df[col] = restored[:, i]
            return self
        else:
            # Nếu user muốn trả về bản khôi phục
            output = input_df.copy()
            for i, col in enumerate(cols_to_inverse):
                output[col] = restored[:, i]
            return output

    # Xuất dữ liệu
    def save(self, df_path, encoders_path):
        """
        Lưu DataPreprocessor thành 2 file CSV:
        - df_path      : CSV chứa dataframe đã xử lý
        - encoders_path: CSV trình bày encoders/mapping
        """
        try:
            # Lưu dataframe
            self.df.to_csv(df_path, index=False)
            print(f"✔ Đã lưu dataframe tại: {df_path}")

            # Lưu encoders
            encoders_list = []
            for col, enc in self.encoders.items():
                if isinstance(enc, dict):
                    # custom mapping
                    for key, val in enc.items():
                        encoders_list.append([col, key, val])
                elif isinstance(enc, LabelEncoder):
                    for i, class_ in enumerate(enc.classes_):
                        encoders_list.append([col, class_, i])
                elif isinstance(enc, OneHotEncoder):
                    for i, class_ in enumerate(enc.categories_[0]):
                        encoders_list.append([col, class_, i])
                else:
                    continue

            enc_df = pd.DataFrame(encoders_list, columns=["column", "category", "encoded"])
            enc_df.to_csv(encoders_path, index=False)
            print(f"Đã lưu encoders tại: {encoders_path}")

        except Exception as e:
            print("Không thể lưu dữ liệu:", e)

    # Lấy dataframe
    def get_df(self):
        return self.df