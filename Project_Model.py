import pandas as pd
import numpy as np
import optuna
import seaborn as sns
import matplotlib.pyplot as plt
import seaborn as sns  
import os
import random
import logging
import joblib 

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import cross_val_score
from sklearn.impute import SimpleImputer

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR

from xgboost import XGBRegressor

# Cấu hình logging cơ bản
import logging
logging.basicConfig(
    filename="training.log",
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

class ModelTrainer:
    """
    Lớp quản lý quy trình huấn luyện, tối ưu hóa và đánh giá các mô hình hồi quy (Regression).

    Attributes:
        test_size (float): Tỷ lệ dữ liệu dành cho tập kiểm tra (test set).
        random_state (int): Hạt giống ngẫu nhiên để đảm bảo tính tái lập.
        df (pd.DataFrame): DataFrame chứa dữ liệu gốc.
        X (pd.DataFrame): DataFrame chứa các đặc trưng (features).
        y (pd.Series): Series chứa biến mục tiêu (target).
        models (dict): Từ điển lưu trữ các mô hình đã huấn luyện.
        results (pd.DataFrame): Bảng lưu trữ kết quả đánh giá (RMSE, MAE, R2).
        best_model (object): Mô hình có hiệu suất tốt nhất sau khi so sánh.vv
    """

    def __init__(self, test_size=0.2, random_state=42):
        """
        Khởi tạo đối tượng ModelTrainer.

        Args:
            test_size (float, optional): Tỷ lệ chia tập test (mặc định 0.2).
            random_state (int, optional): Seed cho các hàm ngẫu nhiên (mặc định 42).
        """
        self.test_size = test_size
        self.random_state = random_state

        # --- REPRODUCIBILITY ---
        np.random.seed(random_state)
        random.seed(random_state)
        os.environ["PYTHONHASHSEED"] = str(random_state)

        print(f"[INFO] Reproducibility enabled — seed = {random_state}")

        self.df = None
        self.X = None
        self.y = None

        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

        self.models = {}              # lưu mô hình đã train
        self.results = pd.DataFrame(columns=["model", "RMSE_Test", "R2_Test"]) # lưu kết quả RMSE/MAE
        self._best_model = None
        self._best_model_name = None

    # ---------------------- DATA METHODS ----------------------
    def load_data(self, file_path, target_column):
        """
        Đọc dữ liệu từ file CSV, xử lý giá trị thiếu và mã hóa biến phân loại.

        Args:
            file_path (str): Đường dẫn đến file CSV dữ liệu.
            target_column (str): Tên cột mục tiêu (nhãn cần dự đoán).

        Returns:
            None: Cập nhật trực tiếp vào self.df, self.X, self.y.
        """
        self.df = pd.read_csv(file_path)
        self.X = self.df.drop(columns=[target_column])
        self.y = self.df[target_column]

        # Encode categorical trước
        self.X = pd.get_dummies(self.X, drop_first=True)

        # Imputer: điền giá trị trung bình cho cột số
        imputer = SimpleImputer(strategy="mean")
        self.X = pd.DataFrame(imputer.fit_transform(self.X), columns=self.X.columns)

    # ---------------------------------------------------------------
    def split_data(self):
        """
        Chia dữ liệu thành tập huấn luyện (train) và tập kiểm tra (test).

        Sử dụng test_size và random_state đã khởi tạo.
        """
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y,
            test_size=self.test_size,
            random_state=self.random_state
        )
        print(">>> Đã chia train/test.")

    # ---------------------- TRAIN & EVALUATE ----------------------
    def train_model(self, model, name):
        """
        Huấn luyện một mô hình cụ thể.

        Args:
            model (sklearn/xgboost model): Đối tượng mô hình chưa huấn luyện.
            name (str): Tên định danh cho mô hình (ví dụ: 'LinearRegression').

        Returns:
            object: Mô hình đã được huấn luyện (fitted model).
        """
        model.fit(self.X_train, self.y_train)
        self.models[name] = model
        print(f">>> Huấn luyện mô hình {name} xong.")
        return model

    # ---------------------------------------------------------------
    def evaluate_model(self, name, model):
        """
        Đánh giá hiệu suất mô hình trên tập Train và Test.

        Args:
            name (str): Tên mô hình.
            model (object): Mô hình đã huấn luyện.

        Returns:
            dict: Dictionary chứa các chỉ số RMSE, MAE, R2 cho Test và RMSE cho Train.
        """
        # Dự đoán trên Tập Kiểm tra (Test) 
        y_pred_test = model.predict(self.X_test)

        # Dự đoán trên Tập Huấn luyện (Train)
        y_pred_train = model.predict(self.X_train)

        metrics = self.compute_metrics(
            self.y_test, y_pred_test,
            y_train=self.y_train,
            y_pred_train=y_pred_train
        )

        return {
            "model": name,
            **metrics
        }

    # ---------------------- STATIC METHODS ----------------------
    @staticmethod
    def compute_metrics(y_test, y_pred_test, y_train=None, y_pred_train=None):
        """
        Tính toán các chỉ số đánh giá cho mô hình Hồi quy (RMSE, MAE, R2).

        Args:
            y_test (array): Giá trị thực tế của tập kiểm tra.
            y_pred_test (array): Giá trị dự đoán trên tập kiểm tra.
            y_train (array, optional): Giá trị thực tế của tập huấn luyện.
            y_pred_train (array, optional): Giá trị dự đoán trên tập huấn luyện.

        Returns:
            dict: Từ điển chứa các chỉ số: RMSE_Test, MAE_Test, R2_Test 
                  (và RMSE_Train nếu cung cấp).
        """ 
        rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))
        mae_test = mean_absolute_error(y_test, y_pred_test)
        r2_test = r2_score(y_test, y_pred_test)

        metrics = {
            "RMSE_Test": rmse_test,
            "MAE_Test": mae_test,
            "R2_Test": r2_test,
        }

        if y_train is not None:
            metrics["RMSE_Train"] = np.sqrt(mean_squared_error(y_train, y_pred_train))

        return metrics

    @staticmethod
    def _plot_comparison_metrics(results: pd.DataFrame, folder: str):
        """
        [STATIC METHOD] Vẽ và lưu biểu đồ so sánh RMSE và R2 giữa các mô hình.

        Args:
            results (pd.DataFrame): DataFrame chứa kết quả đánh giá của các mô hình.
            folder (str): Thư mục để lưu file ảnh.
        """
        # --- BIỂU ĐỒ 1: SO SÁNH RMSE VÀ R2 (Barplot) ---
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # RMSE Plot
        sns.barplot(x="RMSE_Test", y="model", data=results, ax=axes[0], palette="viridis")
        axes[0].set_title("So sánh RMSE (Thấp hơn là tốt hơn)")
        axes[0].set_xlabel("RMSE")

        # R2 Plot
        sns.barplot(x="R2_Test", y="model", data=results, ax=axes[1], palette="magma")
        axes[1].set_title("So sánh R2 Score (Cao hơn là tốt hơn)")
        axes[1].set_xlabel("R2 Score")
        axes[1].set_xlim(0, 1)

        plt.tight_layout()
        file_path = os.path.join(folder, "comparison_metrics.png")
        plt.savefig(file_path)
        plt.close()
        print(f">>> Đã lưu biểu đồ so sánh metrics tại: {file_path}")

    @staticmethod
    def _plot_detailed_analysis(y_test: pd.Series, y_pred: np.ndarray, model_name: str, folder: str):
        """
        [STATIC METHOD] Vẽ và lưu biểu đồ Actual vs Predicted và Residuals Distribution.

        Args:
            y_test (pd.Series): Giá trị thực tế của tập test.
            y_pred (np.ndarray): Giá trị dự đoán trên tập test.
            model_name (str): Tên mô hình đang phân tích.
            folder (str): Thư mục để lưu file ảnh.
        """ 
        residuals = y_test - y_pred

        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        # Subplot 1: Actual vs Predicted
        sns.scatterplot(x=y_test, y=y_pred, ax=axes[0], alpha=0.6, color='blue')
        
        # Vẽ đường chéo đỏ (kỳ vọng lý tưởng)
        min_val = min(y_test.min(), y_pred.min())
        max_val = max(y_test.max(), y_pred.max())
        axes[0].plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
        
        axes[0].set_title(f"{model_name}: Actual vs Predicted")
        axes[0].set_xlabel("Giá Thực tế")
        axes[0].set_ylabel("Giá Dự đoán")

        # Subplot 2: Residual Distribution
        sns.histplot(residuals, kde=True, ax=axes[1], color='purple')
        axes[1].axvline(0, color='r', linestyle='--')
        axes[1].set_title(f"{model_name}: Phân phối Sai số (Residuals)")
        axes[1].set_xlabel("Sai số (Thực tế - Dự đoán)")

        plt.tight_layout()
        file_name = f"analysis_{model_name}.png"
        file_path = os.path.join(folder, file_name)
        plt.savefig(file_path)
        plt.close()
        print(f">>> Đã lưu biểu đồ phân tích cho {model_name} tại: {file_path}")

    # ---------------------- OPTUNA ----------------------
    def objective(self, trial, model_name):
        """
        Hàm mục tiêu cho Optuna để tối ưu hóa siêu tham số.

        Args:
            trial (optuna.trial.Trial): Đối tượng trial của Optuna.
            model_name (str): Tên loại mô hình ('RandomForest' hoặc 'XGBoost').

        Returns:
            float: Giá trị RMSE (trung bình qua Cross-Validation) cần tối thiểu hóa.
        """
        # 1. Định nghĩa không gian tìm kiếm tham số (Search Space)
        if model_name == "RandomForest":
            params = {
                # n_estimators: Số lượng cây (Tìm kiếm số nguyên từ 50 đến 200)
                "n_estimators": trial.suggest_int("n_estimators", 50, 150, step=25),
                # max_depth: Độ sâu tối đa của cây
                "max_depth": trial.suggest_int("max_depth", 5, 20),
                # min_samples_split: Số lượng mẫu tối thiểu để chia nút
                "min_samples_split": trial.suggest_int("min_samples_split", 5, 30),
                "min_samples_leaf": trial.suggest_int("min_samples_leaf", 5, 20),
                "random_state": self.random_state
            }
            model = RandomForestRegressor(**params)

        elif model_name == "XGBoost":
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 50, 200),
                "max_depth": trial.suggest_int("max_depth", 3, 10),
                # learning_rate: Tỷ lệ học (Tìm kiếm theo phân phối log-uniform)
                "learning_rate": trial.suggest_loguniform("learning_rate", 1e-3, 5e-2),
                "subsample": trial.suggest_uniform("subsample", 0.6, 0.9),
                "colsample_bytree": trial.suggest_uniform("colsample_bytree", 0.6, 0.9),
                "reg_alpha": trial.suggest_loguniform("reg_alpha", 1e-8, 1.0),
                "reg_lambda": trial.suggest_loguniform("reg_lambda", 1e-8, 1.0),
                "random_state": self.random_state,
                "verbosity": 0
            }
            model = XGBRegressor(**params)
        else:
            # Không hỗ trợ mô hình này trong objective
            return np.inf

        # 2. Sử dụng Cross-Validation để đánh giá hiệu suất
        # scoring='neg_mean_squared_error' vì chúng ta muốn tối ưu hóa RMSE
        score = cross_val_score(
            model, self.X_train, self.y_train,
            cv=3,
            scoring='neg_mean_squared_error',
            n_jobs=-1
        )

        # 3. Trả về RMSE (Optuna tìm kiếm giá trị nhỏ nhất)
        # Phải lấy căn bậc hai và đảo dấu của điểm số CV (score là neg_MSE)
        rmse = np.sqrt(-score.mean())
        return rmse

    # ---------------------------------------------------------------
    def optimize_params(self, model_name, n_trials=20):
        """
        Chạy Optuna để tìm bộ siêu tham số tối ưu và huấn luyện lại mô hình.

        Args:
            model_name (str): Tên mô hình cần tối ưu.
            n_trials (int, optional): Số lượng lần thử nghiệm của Optuna. Defaults to 20.

        Returns:
            object: Mô hình tốt nhất sau khi tối ưu và huấn luyện lại trên toàn bộ tập train.
        """
        if model_name not in ["RandomForest", "XGBoost"]:
            print(f"Không hỗ trợ tối ưu Optuna cho {model_name}.")
            return None

        print(f"\n{'='*30}")
        print(f">>> Đang tối ưu siêu tham số cho {model_name} bằng Optuna...")
        print(f"    Sử dụng {n_trials} lần thử (trials)...")

        # Tạo Study của Optuna
        # direction="minimize" vì chúng ta muốn giảm thiểu RMSE
        study = optuna.create_study(direction="minimize")

        # Khởi chạy quá trình tối ưu
        # Truyền model_name vào objective bằng cách sử dụng lambda
        study.optimize(
            lambda trial: self.objective(trial, model_name),
            n_trials=n_trials,
            n_jobs=1,
            show_progress_bar=True
        )

        # Lấy tham số và mô hình tốt nhất
        best_params = study.best_params
        print(f"\n>>> Tham số tốt nhất cho {model_name}: {best_params}")
        print(f">>> Best CV RMSE: {study.best_value:.2f}")

        # 1. Huấn luyện mô hình cuối cùng với tham số tốt nhất
        if model_name == "RandomForest":
            best_model = RandomForestRegressor(**best_params, random_state=self.random_state)
        elif model_name == "XGBoost":
            best_model = XGBRegressor(**best_params, random_state=self.random_state, verbosity=0)

        best_model.fit(self.X_train, self.y_train)

        # 2. Cập nhật kết quả vào Class
        self.models[model_name] = best_model

        # Xóa kết quả đánh giá cũ của mô hình này (nếu có trong results)
        if not self.results.empty:
           self.results = self.results[self.results["model"] != model_name].copy()

        # Đánh giá mô hình tối ưu trên tập Test và thêm vào results
        metrics = self.evaluate_model(model_name, best_model)
        self.results = pd.concat([self.results, pd.DataFrame([metrics])], ignore_index=True)

        print(f"    RMSE Test (Mô hình Tối ưu): {metrics['RMSE_Test']:.2f}")

        return best_model

    # ---------------------- RUN ----------------------
    def run_all_models(self):
        """
        Quy trình chính: Chạy LinearRegression, tối ưu RF & XGBoost, so sánh và lưu kết quả.

        Returns:
            pd.DataFrame: Bảng tổng hợp kết quả đánh giá các mô hình.
        """
        candidate_models = {
            "LinearRegression": LinearRegression(),
            "RandomForest": RandomForestRegressor(random_state=self.random_state),
            "XGBoost": XGBRegressor(random_state=self.random_state, verbosity=0),
        }

        results_list = []
        self.models = {}  # Reset models trước khi train

        for name, model in candidate_models.items():
            print("\n============================")
            print(f"Đang huấn luyện mô hình: {name}")
            logging.info(f"Training model: {name}")

            if name in ["RandomForest", "XGBoost"]:
                # Tối ưu siêu tham số
                model = self.optimize_params(name)
            else:
                # LinearRegression → train trực tiếp
                self.train_model(model, name)

            # Đánh giá mô hình
            metrics = self.evaluate_model(name, model)
            results_list.append(metrics)

            # Lưu vào self.models
            self.models[name] = model
            logging.info(f"Model {name} metrics: {metrics}")

        self.results = pd.DataFrame(results_list)

        # Sort results
        self.results = self.results.sort_values(
            by=["RMSE_Test", "R2_Test"], ascending=[True, False]
        )

        best_name = self.best_model_name
        best_row = self.results.iloc[0]

        self.save_model(best_name)

        # Hiển thị bảng kết quả
        print("\n" + "=" * 50)
        print(">>> BẢNG KẾT QUẢ ĐÁNH GIÁ CUỐI CÙNG <<<")
        print(self.results.to_markdown(index=False, floatfmt=".2f"))
        print("=" * 50)

        # In thông tin mô hình tốt nhất
        best_metrics = self.results[self.results['model'] == self.best_model_name].iloc[0]
        print(f"  MÔ HÌNH ĐƯỢC CHỌN: {self.best_model_name}")
        print(f"   - RMSE Test: {best_metrics['RMSE_Test']:.2f}")
        print(f"   - R2 Test: {best_metrics['R2_Test']:.4f}")
        print(f"   - RMSE Train: {best_metrics['RMSE_Train']:.2f}")

        # Lưu kết quả CSV
        self.save_results("experiment_results.csv")

        # Vẽ và lưu biểu đồ
        print("\n>>> ĐANG VẼ BIỂU ĐỒ SO SÁNH...")
        self.visualize_results()

        return self.results

    # ---------------------- PROPERTY ----------------------
    @property
    def best_model_name(self):
        """
        [PROPERTY] Trả về tên mô hình tốt nhất dựa trên RMSE_Test thấp nhất.

        Returns:
            str or None: Tên mô hình (ví dụ: 'XGBoost') hoặc None nếu chưa có kết quả.
        """
        if self.results.empty:
            return None
        return self.results.sort_values("RMSE_Test").iloc[0]["model"]

    @property
    def best_model(self):
        """
        [PROPERTY] Trả về đối tượng mô hình (fitted model) có hiệu suất tốt nhất.

        Returns:
            object or None: Đối tượng mô hình đã được huấn luyện hoặc None.
        """
        name = self.best_model_name
        if name:
            return self.models.get(name)
        return None

    # ---------------------------------------------------------------
    def save_model(self, model_name, file_path=None):
        """
        Lưu mô hình đã train ra file .pkl bằng joblib.
        """
        if model_name not in self.models:
            print(f"Không tìm thấy mô hình {model_name} để lưu.")
            return

        if file_path is None:
            file_path = f"best_model_{model_name}.pkl"

        joblib.dump(self.models[model_name], file_path)
        print(f">>> Mô hình {model_name} đã được lưu tại: {file_path}")
        logging.info(f"Saved model {model_name} to {file_path}")

    # ---------------------------------------------------------------
    def load_model(self, file_path):
        """
        Nạp mô hình từ file .pkl bằng joblib.
        """
        model = joblib.load(file_path)
        print(f">>> Mô hình đã được nạp từ: {file_path}")
        logging.info(f"Loaded model from {file_path}")
        return model

    # ---------------------------------------------------------------
    def save_results(self, file_path="experiment_results.csv"):
        """
        Lưu DataFrame kết quả đánh giá tất cả mô hình ra file CSV.
        """
        if self.results.empty:
            print("Không có kết quả để lưu.")
            return

        self.results.to_csv(file_path, index=False)
        print(f">>> Kết quả đã được lưu tại: {file_path}")
        logging.info(f"Saved experiment results to {file_path}")

    # ---------------------------------------------------------------
    def visualize_results(self, folder='artifacts'):
        """
        Vẽ và lưu các biểu đồ đánh giá: So sánh Metrics, Actual vs Predicted và Residuals.
        """
        # ... (Phần kiểm tra self.results.empty và tạo folder giữ nguyên) ...
        if self.results.empty:
            print("Chưa có kết quả để vẽ biểu đồ.")
            return

        if not os.path.exists(folder):
            os.makedirs(folder, exist_ok=True)

        sns.set(style="whitegrid")
        
        # --- GỌI STATIC METHOD VẼ BIỂU ĐỒ 1 ---
        ModelTrainer._plot_comparison_metrics(self.results, folder)
        
        # --- GỌI STATIC METHOD VẼ BIỂU ĐỒ 2 & 3 ---
        for name, model in self.models.items():
            # Bước này vẫn phải nằm trong phương thức cá thể vì cần truy cập self.X_test
            y_pred = model.predict(self.X_test)
            
            # Gọi phương thức tĩnh mới với các dữ liệu đã tính toán được truyền vào
            ModelTrainer._plot_detailed_analysis(
                self.y_test, # Giá trị thực tế (pd.Series)
                y_pred,     # Giá trị dự đoán (np.ndarray)
                name,       # Tên mô hình (str)
                folder      # Thư mục lưu (str)
            )

def main():
    # 1. Đường dẫn tới file CSV và tên cột target
    file_path = "Car details v3.csv"
    target_column = "selling_price"

    # 2. Khởi tạo đối tượng ModelTrainer
    trainer = ModelTrainer(test_size=0.2, random_state=42)

    # 3. Nạp dữ liệu
    trainer.load_data(file_path, target_column)

    # 4. Chia train/test
    trainer.split_data()

    # 5. Chạy tất cả các mô hình đã định nghĩa
    trainer.run_all_models()


# ---------------------------------------------------------------
if __name__ == "__main__":
    main()