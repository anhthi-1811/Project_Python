import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import os
import shap
from sklearn.inspection import PartialDependenceDisplay


class Visualizer:
    """
    Class xử lý trực quan hóa dữ liệu và kết quả mô hình
    Vẽ tất cả các biểu đồ: EDA, đánh giá mô hình, giải thích mô hình
    1. EDA: Phân phối biến số, biến phân loại, correlation matrix
    2. Đánh giá mô hình: So sánh RMSE/R2, Actual vs Predicted, Residuals
    3. Giải thích mô hình: Feature Importance, SHAP, PDP
    """

    def __init__(self, data, target_col, output_dir='plots/'):
        """
        Khởi tạo Visualizer.
        
        Args:
            data (pd.DataFrame): Dữ liệu gốc (chưa qua xử lý) để vẽ EDA.
            target_col (str): Tên cột mục tiêu (biến phụ thuộc).
            output_dir (str): Đường dẫn thư mục lưu ảnh.
        """
        self.data = data
        self.target_col = target_col
        self.output_dir = output_dir
        # Tạo thư mục lưu ảnh nếu chưa có
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    # ===========================================================
    # --- NHÓM 1: TRỰC QUAN HÓA DỮ LIỆU (EDA) ---
    # Mục đích: Hiểu phân phối dữ liệu, mối quan hệ giữa các biến trước khi mô hình hóa
    # ===========================================================

    def plot_numerical_distribution(self, numerical_cols):
        """Vẽ biểu đồ phân phối (Histogram + KDE) cho các biến số"""

        n = len(numerical_cols)
        # Tính số hàng cần thiết (2 cột mỗi hàng)
        rows = (n + 1) // 2

        fig, axes = plt.subplots(rows, 2, figsize=(14, rows * 4))
        axes = axes.flatten()

        for i, col in enumerate(numerical_cols):
            # Vẽ Histogram kèm đường mật độ (KDE)
            sns.histplot(self.data[col], kde=True, ax=axes[i], color='skyblue')
            axes[i].set_title(f"Distribution of {col}")
            axes[i].set_xlabel(col)
            axes[i].set_ylabel("Tần suất")

        # Xoá các subplot thừa nếu số biến lẻ
        for j in range(i + 1, len(axes)):
            fig.delaxes(axes[j])
        
        plt.tight_layout()

        # Lưu file
        file_name = "numerical_distribution_all.png"
        file_path = os.path.join(self.output_dir, file_name)
        plt.savefig(file_path, dpi=300)
        plt.close()
        print(f">>> [EDA] Đã lưu phân phối biến số: {file_name}")

    def plot_correlation(self, numerical_cols):
        """
        Vẽ 2 biểu đồ:
        1) Full Correlation Matrix
        2) Correlation với Target
        Giúp thấy tổng quan và tập trung vào mối quan hệ với biến mục tiêu.
        """
        cols = numerical_cols.copy()
        if self.target_col not in cols:
            cols.append(self.target_col)

        corr = self.data[cols].corr()

        # Tự tạo figure với 2 subplot
        fig, axes = plt.subplots(1, 2, figsize=(18, 7))

        # --- (1) FULL CORRELATION MATRIX ---
        sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm', ax=axes[0])
        axes[0].set_title("Full Correlation Matrix")

        # --- (2) TARGET CORRELATION ONLY ---
        target_corr = corr[self.target_col].drop(self.target_col).sort_values(ascending=False)

        sns.heatmap(target_corr.to_frame(), annot=True, cmap="coolwarm", ax=axes[1])
        axes[1].set_title(f"Correlation with Target: {self.target_col}")

        plt.tight_layout()

        file_name = "correlation_full_and_target.png"
        file_path = os.path.join(self.output_dir, file_name)
        plt.savefig(file_path, dpi=300)
        plt.close()

        print(f">>> [EDA] Đã lưu biểu đồ tương quan kép: {file_name}")

    # Biểu đồ cột cho biến phân loại
    def plot_categorical(self, cat_cols):
        """
        Vẽ biểu đồ cột (Countplot) cho các biến phân loại.
        Giúp xem mức độ cân bằng của dữ liệu.
        """
        for col in cat_cols:
            plt.figure(figsize=(8, 6))
            # Sắp xếp theo số lượng giảm dần cho dễ nhìn
            order = self.data[col].value_counts().index
            sns.countplot(x=self.data[col], order=order, palette="viridis")

            plt.title(f'Phân phối của biến phân loại: {col}')
            plt.xticks(rotation=45) # Xoay nhãn trục X nếu dài
            plt.tight_layout()

            file_name = f"categorical_{col}_distribution.png"
            file_path = os.path.join(self.output_dir, file_name)
            plt.savefig(file_path, dpi=300)
            plt.close()
        print(f">>> [EDA] Đã lưu {len(cat_cols)} biểu đồ biến phân loại.")

    def plot_target_distribution(self):
        """ Vẽ phân phối riêng của biến mục tiêu (Target)."""
        plt.figure(figsize=(10, 6))
        sns.histplot(self.data[self.target_col], kde=True, color='salmon')
        plt.title(f'Phân phối của {self.target_col}')
        plt.tight_layout()
        file_name = "target_distribution.png"
        file_path = os.path.join(self.output_dir, file_name)
        plt.savefig(file_path, dpi=300)
        plt.close()
        print(f">>> [EDA] Đã lưu phân phối Target: {file_name}")

    # ===========================================================
    # --- NHÓM 2: ĐÁNH GIÁ MÔ HÌNH (Regression) ---
    # Mục đích: So sánh hiệu suất và kiểm tra độ tin cậy của dự đoán.
    # ===========================================================
    def plot_comparison_metrics(self, results: pd.DataFrame):
        """
        Vẽ biểu đồ so sánh tổng quan các mô hình dựa trên RMSE và R2.
        Args:
            results (pd.DataFrame): Bảng kết quả chứa cột 'model', 'RMSE_Test', 'R2_Test'.
        """
        if results is None or results.empty: return

        # SO SÁNH RMSE VÀ R2 (Barplot) 
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # Biểu đồ 1: So sánh RMSE (Càng thấp càng tốt)
        sns.barplot(x="RMSE_Test", y="model", data=results, ax=axes[0], palette="viridis")
        axes[0].set_title("So sánh RMSE (Thấp hơn là tốt hơn)")
        axes[0].set_xlabel("RMSE")

        # Biểu đồ 2: So sánh R2 Score (Càng cao càng tốt)
        sns.barplot(x="R2_Test", y="model", data=results, ax=axes[1], palette="magma")
        axes[1].set_title("So sánh R2 Score (Cao hơn là tốt hơn)")
        axes[1].set_xlabel("R2 Score")
        axes[1].set_xlim(0, 1.05) # Giới hạn trục R2 max là 1

        plt.tight_layout()
        file_name = "comparison_metrics.png"
        file_path = os.path.join(self.output_dir, file_name)
        plt.savefig(file_path, dpi=300)
        plt.close()
        print(f">>> [EVAL] Đã lưu biểu đồ so sánh Model: {file_name}")

    def plot_model_performance(self, y_true, y_pred, model_name):
        """
        Vẽ biểu đồ đánh giá chi tiết cho MỘT mô hình cụ thể:
        1. Actual vs Predicted: So khớp thực tế và dự đoán (các điểm càng gần đường đỏ càng tốt).
        2. Residual Distribution: Phân phối phần dư/sai số (nên gần chuẩn, hình chuông).
        """
        
        # Tính phần dư (Sai số)
        residuals = y_true - y_pred

        # Tạo khung hình với 2 biểu đồ con (1 hàng, 2 cột)
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        # --- SUBPLOT 1: Actual vs Predicted ---
        sns.scatterplot(x=y_true, y=y_pred, ax=axes[0], alpha=0.6, color='blue')
        
        # Vẽ đường chéo đỏ (Kỳ vọng lý tưởng y=x)
        min_val = min(y_true.min(), y_pred.min())
        max_val = max(y_true.max(), y_pred.max())
        axes[0].plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
        
        axes[0].set_title(f"{model_name}: Actual vs Predicted", fontsize=14)
        axes[0].set_xlabel("Giá Thực tế")
        axes[0].set_ylabel("Giá Dự đoán")

        # --- SUBPLOT 2: Residual Distribution (Histogram) ---
        # Dùng histplot để xem phân phối lỗi có chuẩn (hình chuông) không
        sns.histplot(residuals, kde=True, ax=axes[1], color='purple')
        axes[1].axvline(0, color='r', linestyle='--', lw=2) # Đường 0 tham chiếu
        
        axes[1].set_title(f"{model_name}: Phân phối Sai số (Residuals)", fontsize=14)
        axes[1].set_xlabel("Sai số (Thực tế - Dự đoán)")
        axes[1].set_ylabel("Tần suất")

        # Tinh chỉnh layout
        plt.tight_layout()

        # Lưu file chuẩn
        file_name = f"performance_{model_name}.png"
        file_path = os.path.join(self.output_dir, file_name)
        
        plt.savefig(file_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f">>> [EVAL] Đã lưu đánh giá chi tiết {model_name}: {file_name}")

    # ===========================================================
    # --- NHÓM 3: GIẢI THÍCH MÔ HÌNH (XAI - Explainable AI) ---
    # Mục đích: Hiểu tại sao mô hình đưa ra dự đoán đó.
    # ===========================================================

    def plot_feature_importance(self, model, feature_names, top_k=15):
        """
        Vẽ biểu đồ mức độ quan trọng của các đặc trưng (Feature Importance).
        Giúp trả lời câu hỏi: Yếu tố nào ảnh hưởng nhất đến giá xe?
        """
        importances = None
        
        # Lấy độ quan trọng tùy theo loại mô hình
        # Trường hợp 1: Tree models (RF, XGB)
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
        # Trường hợp 2: Linear Regression (dùng coef_)
        elif hasattr(model, 'coef_'):
            importances = np.abs(model.coef_) # Lấy trị tuyệt đối
        else:
            print("[WARN] Model không hỗ trợ trích xuất Feature Importance.")
            return

        if len(feature_names) != len(importances):
            # Xử lý trường hợp one-hot encoding làm thay đổi số lượng feature
            print("[WARN] Số lượng tên feature và độ quan trọng không khớp.")
            return
        
        # Sắp xếp giảm dần và lấy top_k feature quan trọng nhất
        idx = np.argsort(importances)[::-1][:top_k]
        
        plt.figure(figsize=(12, 8))
        sns.barplot(x=importances[idx], y=np.array(feature_names)[idx])
        plt.title(f"{top_k} Feature Importance")
        plt.xlabel("Importance Score")
        plt.tight_layout()
        file_name = "feature_importance.png"
        file_path = os.path.join(self.output_dir, file_name)
        plt.savefig(file_path, dpi=300)
        plt.close()
        print(f">>> [XAI] Đã lưu Feature Importance: {file_name}")

    def plot_shap_summary(self, model, X_train):
        """
        Vẽ biểu đồ SHAP Summary (Bee swarm plot).
        Cung cấp cái nhìn sâu hơn Feature Importance:
        - Màu đỏ: Giá trị feature cao.
        - Màu xanh: Giá trị feature thấp.
        - Vị trí trái/phải: Tác động tiêu cực/tích cực lên giá dự đoán.
        """
        try:
            # Dùng Explainer chung (tự động nhận diện Tree hoặc Linear)
            explainer = shap.Explainer(model, X_train)

            # Tính SHAP values (chỉ lấy 500 mẫu để chạy nhanh nếu dữ liệu lớn)
            sample_data = X_train.iloc[:500] if len(X_train) > 500 else X_train
            shap_values = explainer(sample_data)
            
            plt.figure(figsize=(10, 6))

            shap.summary_plot(shap_values, sample_data, show=False)
            plt.tight_layout()
            file_name = "shap_summary.png"
            file_path = os.path.join(self.output_dir, file_name)
            plt.savefig(file_path, dpi=300)
            plt.close()
            print(f">>> [XAI] Đã lưu SHAP Summary: {file_name}")
        except Exception as e:
            print(f"[ERROR] Không thể vẽ SHAP: {e}")

    def plot_pdp(self, model, X, feature):
        """
        Vẽ biểu đồ sự phụ thuộc một phần (Partial Dependence Plot - PDP).
        Cho thấy mối quan hệ giữa MỘT đặc trưng cụ thể và giá dự đoán 
        (ví dụ: xe càng cũ giá càng giảm như thế nào, tuyến tính hay phi tuyến).
        """
        try:
            fig, ax = plt.subplots(figsize=(8, 6))
            PartialDependenceDisplay.from_estimator(model, X, [feature], ax=ax)
            plt.title(f"Partial Dependence Plot: {feature}")
            plt.tight_layout()
            file_name = f"pdp_{feature}.png"
            file_path = os.path.join(self.output_dir, file_name)
            plt.savefig(file_path, dpi=300)
            plt.close()
            print(f">>> [XAI] Đã lưu PDP cho {feature}: {file_name}")
        except Exception as e:
            print(f"[ERROR] Lỗi vẽ PDP cho {feature}: {e}")

