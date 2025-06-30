from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
import numpy as np

# Khởi tạo ứng dụng FastAPI
app = FastAPI(
    title="Dermatology Diagnosis API",
    description="An API to predict dermatology diseases using a Scikit-learn pipeline.",
    version="1.0.0"
)

# Tải pipeline đã được huấn luyện
# File này phải nằm cùng thư mục với main.py
try:
    pipeline = joblib.load('best_dermatology_pipeline.pkl')
    print("Model pipeline loaded successfully!")
except FileNotFoundError:
    print("Error: Model file not found. Make sure 'best_dermatology_pipeline.pkl' is in the same directory.")
    pipeline = None

# Định nghĩa cấu trúc dữ liệu đầu vào cho API bằng Pydantic
# Gồm 33 thuộc tính lâm sàng và mô bệnh học (trừ 'family_history' và 'age' là số nguyên)
class ClinicalData(BaseModel):
    erythema: int
    scaling: int
    definite_borders: int
    itching: int
    koebner_phenomenon: int
    polygonal_papules: int
    follicular_papules: int
    oral_mucosal_involvement: int
    knee_and_elbow_involvement: int
    scalp_involvement: int
    family_history: int  # 0 or 1
    melanin_incontinence: int
    eosinophils_in_the_infiltrate: int
    pnl_infiltrate: int
    fibrosis_of_the_papillary_dermis: int
    exocytosis: int
    acanthosis: int
    hyperkeratosis: int
    parakeratosis: int
    clubbing_of_the_rete_ridges: int
    elongation_of_the_rete_ridges: int
    thinning_of_the_suprapapillary_epidermis: int
    spongiform_pustule: int
    munro_microabcess: int
    focal_hypergranulosis: int
    disappearance_of_the_granular_layer: int
    vacuolisation_and_damage_of_basal_layer: int
    spongiosis: int
    saw_tooth_appearance_of_retes: int
    follicular_horn_plug: int
    perifollicular_parakeratosis: int
    inflammatory_mononuclear_infiltrate: int
    band_like_infiltrate: int
    age: int # Tuổi của bệnh nhân

# Endpoint gốc để kiểm tra API có hoạt động không
@app.get("/")
def read_root():
    return {"message": "Welcome to the Dermatology Diagnosis API. Use the /predict endpoint to make a diagnosis."}

# Endpoint để dự đoán
@app.post("/predict")
def predict(data: ClinicalData):
    if pipeline is None:
        return {"error": "Model is not loaded. Please check the server logs."}

    # Chuyển dữ liệu từ Pydantic model thành dictionary
    input_data = data.dict()
    # Chuyển dictionary thành DataFrame của Pandas vì pipeline của chúng ta cần đầu vào là DataFrame
    # Lưu ý thứ tự các cột phải chính xác như lúc huấn luyện
    input_df = pd.DataFrame([input_data])

    # Thực hiện dự đoán
    try:
        prediction_index = pipeline.predict(input_df)[0]
        prediction_probabilities = pipeline.predict_proba(input_df)[0]

        # Ánh xạ chỉ số dự đoán (0-5) sang tên bệnh thực tế
        disease_mapping = {
            0: "Psoriasis",
            1: "Seborrheic Dermatitis",
            2: "Lichen Planus",
            3: "Pityriasis Rosea",
            4.0: "Chronic Dermatitis",
            5: "Pityriasis Rubra Pilaris"
        }
        
        disease_name = disease_mapping.get(prediction_index, "Unknown Disease")
        confidence = float(prediction_probabilities[prediction_index]) # Chuyển numpy.float64 thành float của Python

        return {
            "predicted_disease": disease_name,
            "predicted_class_index": int(prediction_index),
            "confidence_score": round(confidence, 4)
        }
    except Exception as e:
        return {"error": f"An error occurred during prediction: {str(e)}"}