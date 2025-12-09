import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import os
import nbformat

from Project_Model import ModelTrainer 
from visualization import Visualizer 
from nbconvert.preprocessors import ExecutePreprocessor


def main():
    ### TIỀN XỬ LÝ
    notebook_path = "Data_processing.ipynb"

    with open(notebook_path, encoding="utf-8") as f:
        nb = nbformat.read(f, as_version=4)

    ep = ExecutePreprocessor(timeout=600, kernel_name='python3')
    ep.preprocess(nb, {'metadata': {'path': './'}})

    ### CHẠY MÔ HÌNH VÀ TRỰC QUAN DỮ LIỆU
    # 1. CHẠY MÔ HÌNH (Lấy dữ liệu từ Project_Model)
    file_path = "data_processed.csv"
    target_col = "selling_price"

    # Khởi tạo và chạy quy trình huấn luyện
    trainer = ModelTrainer(test_size=0.2, random_state=42)
    trainer.load_data(file_path, target_col)
    trainer.split_data()
    
    # Bước này sẽ train các model và tìm ra model tốt nhất
    results_df = trainer.run_all_models()
    
    # LẤY CÁC BIẾN QUAN TRỌNG ĐỂ VẼ
    #  Model tốt nhất (để vẽ feature importance/shap)
    best_model = trainer.best_model 
    #  Dữ liệu đã xử lý (để tính SHAP/Feature Importance)
    X_train_processed = trainer.X_train
    
    print(f">>> Model tốt nhất: {trainer.best_model_name}")

    # 2. KHỞI TẠO VISUALIZER
    data = pd.read_csv(file_path)

    viz = Visualizer(data=data, target_col=target_col, output_dir='plots')

    # 3. TRỰC QUAN DỮ LIỆU (EDA)

    # Vẽ biến số (Dùng các cột số sạch như 'mileage', 'torque_value'...)
    # Loại bỏ các cột ID, cột đã chuẩn hóa (_std) hoặc cột label (_label) để biểu đồ đỡ rối
    numerical_cols = [c for c in data.select_dtypes(include=np.number).columns 
                    if not c.endswith(('_std', '_label')) and c != target_col]
    # Loại bỏ target nếu nó nằm trong danh sách input
    if target_col in numerical_cols:
        numerical_cols.remove(target_col)
    
    # Vẽ biến phân loại (Dùng cột gốc dạng chữ như 'fuel', 'seller_type'...)
    # Loại bỏ cột 'name' (quá nhiều giá trị) và 'torque' (đã xử lý thành số)
    categorical_cols = [c for c in data.select_dtypes(include=['object']).columns 
                if c not in ['name', 'torque']]

    print("   - Vẽ phân phối dữ liệu đầu vào...")
    viz.plot_target_distribution()
    viz.plot_numerical_distribution(numerical_cols)              
    viz.plot_categorical(categorical_cols)
    viz.plot_correlation(numerical_cols) 

    # 4. TRỰC QUAN KẾT QUẢ MÔ HÌNH

    print("   - Vẽ đánh giá mô hình...")
     # a) So sánh tổng quan (Bar chart)
    viz.plot_comparison_metrics(results_df)
    
    # b) Đánh giá chi tiết TỪNG MÔ HÌNH
    for name, model in trainer.models.items():
        # Dự đoán trên tập test
        y_pred = model.predict(trainer.X_test)
        # Gọi hàm vẽ (sẽ lưu file performance_LinearRegression.png, performance_XGBoost.png...)
        viz.plot_model_performance(trainer.y_test, y_pred, name)

    # 5. GIẢI THÍCH MÔ HÌNH (Explainable AI)

    print("   - Vẽ Feature Importance & SHAP...")
    
    # Lấy tên feature từ X_train (Dữ liệu đã qua One-Hot Encoding)
    feature_names = X_train_processed.columns.tolist()

    viz.plot_feature_importance(best_model, feature_names)
    viz.plot_shap_summary(best_model, X_train_processed)

    # Partial Dependence Plot (PDP) cho feature quan trọng nhất
    if hasattr(best_model, 'feature_importances_'):
        top_idx = np.argsort(best_model.feature_importances_)[-1]
        top_feature = feature_names[top_idx]
        print(f"     Vẽ PDP cho feature quan trọng nhất: {top_feature}")
        viz.plot_pdp(best_model, X_train_processed, top_feature)

    print(f">>> Hoàn tất! ")

if __name__ == "__main__":
    main()

