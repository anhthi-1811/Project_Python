import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import os
import shap
from sklearn.inspection import PartialDependenceDisplay

import joblib  # Thư viện để load model
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer

class Visualizer:
    def __init__(self, data, target_col, output_dir='plots/'):
        """
        data: DataFrame gốc (để vẽ phân phối đầu vào).
        target_col: Tên cột mục tiêu.
        output_dir: Thư mục lưu ảnh.
        """
        self.data = data
        self.target_col = target_col
        self.output_dir = output_dir
        # Tạo thư mục lưu ảnh nếu chưa có
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    # --- NHÓM 1: TRỰC QUAN HÓA DỮ LIỆU (EDA) ---

    def plot_numerical_distribution(self, numerical_cols):
        """Vẽ phân phối các biến số"""
        n = len(numerical_cols)
        rows = (n + 1) // 2

        fig, axes = plt.subplots(rows, 2, figsize=(14, rows * 4))
        axes = axes.flatten()

        for i, col in enumerate(numerical_cols):
            sns.histplot(self.data[col], kde=True, ax=axes[i])
            axes[i].set_title(f"Distribution of {col}")

        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/numerical_distribution_all.png", dpi=300)
        plt.close()

    def plot_correlation(self, numerical_cols):
        """Vẽ full correlation matrix (Spearman)."""
        cols = numerical_cols.copy()
        if self.target_col in self.data.columns and self.target_col not in cols:
            cols.append(self.target_col)
            
        corr = self.data[cols].corr()
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm')
        plt.title('Correlation Matrix')
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/correlation_matrix.png", dpi=300)
        plt.close()

    # Biểu đồ cột cho biến phân loại
    def plot_categorical(self, cat_cols):
        """Vẽ biểu đồ cột cho các biến phân loại"""
        for col in cat_cols:
            plt.figure(figsize=(8, 6))
            sns.countplot(x=self.data[col])
            plt.title(f'Phân phối của {col}')
            plt.tight_layout()
            plt.savefig(f"{self.output_dir}/{col}_countplot.png", dpi=300)
            plt.close()

    def plot_target_distribution(self):
        """Vẽ phân phối biến mục tiêu (Yêu cầu Input Data) """
        plt.figure(figsize=(8, 6))
        sns.histplot(self.data[self.target_col])
        plt.title(f'Phân phối của {self.target_col}')
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/target_distribution.png", dpi=300)
        plt.close()

    # --- NHÓM 2: ĐÁNH GIÁ MÔ HÌNH (Regression) ---

    def plot_actual_vs_predicted(self, y_true, y_pred, model_name):
        """Vẽ biểu đồ giá trị Thực tế vs Dự đoán"""
        plt.figure(figsize=(8, 6))
        plt.scatter(y_true, y_pred, alpha=0.5, color='blue')
        
        # Vẽ đường chéo đỏ (kỳ vọng lý tưởng)
        min_val = min(y_true.min(), y_pred.min())
        max_val = max(y_true.max(), y_pred.max())
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
        
        plt.xlabel('Thực tế')
        plt.ylabel('Dự đoán')
        plt.title(f'{model_name}: Actual vs Predicted')
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/{model_name}_actual_vs_pred.png", dpi=300)
        plt.close()

    def plot_residuals(self, y_true, y_pred):
        residuals = y_true - y_pred
    
        plt.figure(figsize=(8, 6))
        sns.scatterplot(x=y_pred, y=residuals)
        plt.axhline(0, color='red', linestyle='--')
        plt.xlabel("Predicted")
        plt.ylabel("Residuals")
        plt.title("Residual Plot")
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/residual_plot.png", dpi=300)
        plt.close()

    def plot_model_comparison(self, results_df, metric_name='RMSE_Test'):
        """So sánh các mô hình (Bar chart)"""
        if results_df is None or results_df.empty: return
        
        plt.figure(figsize=(10, 6))
        # Sắp xếp để nhìn rõ hơn
        df_sorted = results_df.sort_values(by=metric_name, ascending=True) 
        sns.barplot(x='model', y=metric_name, data=df_sorted, palette='viridis',hue='model', legend=False)
        plt.xticks(rotation=20)
        plt.title(f'Model Comparison by {metric_name}')
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/model_comparison_{metric_name}.png", dpi=300)
        plt.close()
    
    # --- NHÓM 3: GIẢI THÍCH MÔ HÌNH (Explainable AI) ---

    def plot_feature_importance(self, model, feature_names, top_k=15):
        """Xử lý linh hoạt cho cả Tree-based và Linear models"""
        importances = None
        
        # Trường hợp 1: Tree models (RF, XGB)
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
        # Trường hợp 2: Linear Regression (dùng coef_)
        elif hasattr(model, 'coef_'):
            importances = np.abs(model.coef_) # Lấy trị tuyệt đối
        else:
            print("Model không hỗ trợ trích xuất Feature Importance mặc định.")
            return

        # Sắp xếp và vẽ
        if len(feature_names) != len(importances):
            # Xử lý trường hợp one-hot encoding làm thay đổi số lượng feature
            print("Cảnh báo: Số lượng tên feature và độ quan trọng không khớp.")
            return

        idx = np.argsort(importances)[::-1][:top_k]
        
        plt.figure(figsize=(10, 6))
        sns.barplot(x=importances[idx], y=np.array(feature_names)[idx])
        plt.title("Top Feature Importance")
        plt.xlabel("Importance Score")
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/feature_importance.png", dpi=300)
        plt.close()

    def plot_shap_summary(self, model, X_train):
        """Vẽ SHAP summary (Tự động chọn Explainer phù hợp)"""
        try:
            # Dùng Explainer chung (tự động nhận diện Tree hoặc Linear)
            explainer = shap.Explainer(model, X_train)
            shap_values = explainer(X_train)
            
            plt.figure(figsize=(10, 6))
            # shap_values.values cho object mới của shap
            shap.summary_plot(shap_values, X_train, show=False)
            plt.tight_layout()
            plt.savefig(f"{self.output_dir}/shap_summary.png", dpi=300)
            plt.close()
        except Exception as e:
            print(f"Không thể vẽ SHAP: {e}")

    def plot_pdp(self, model, X, feature):
        """Vẽ Partial Dependence Plot """
        try:
            fig, ax = plt.subplots(figsize=(8, 6))
            PartialDependenceDisplay.from_estimator(model, X, [feature], ax=ax)
            plt.title(f"Partial Dependence Plot: {feature}")
            plt.tight_layout()
            plt.savefig(f"{self.output_dir}/pdp_{feature}.png", dpi=300)
            plt.close()
        except Exception as e:
            print(f"Lỗi vẽ PDP cho {feature}: {e}")

