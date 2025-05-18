import os
import time
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template_string, redirect, url_for
from keras.models import load_model
import joblib
from collections import Counter
from tienxuly import create_input_processor

# ==== CẤU HÌNH ====
MODELS = {
    "cnn": {
        "path": "models/cnn.h5",
        "display_name": "CNN"
    },
    "logistic": {
        "path": "models/logistic_regressor_multi.h5",
        "display_name": "Logistic Regression"
    },
    "lstm": {
        "path": "models/lstm.h5",
        "display_name": "LSTM"
    }
}
SCALER_PATH = "models/scaler.pkl"
UPLOAD_FOLDER = "uploads"

# Thêm mapping tên loại tấn công
ATTACK_TYPES = {
    0: "Normal",
    1: "DoS",
    2: "Probe",
    3: "R2L",
    4: "U2R",
    5: "Backdoor",
    6: "Exploits",
    7: "Analysis",
    8: "Fuzzers",
    9: "Worms",
    10: "Shellcode",
    11: "Generic"
}

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ==== TẢI MODEL & SCALER ====
print("🔄 Đang load models và scaler...")
loaded_models = {}

for model_key, model_info in MODELS.items():
    model_path = model_info["path"]
    if not os.path.exists(model_path):
        print(f"⚠️ Không tìm thấy model {model_key} tại {model_path}, sẽ bỏ qua")
        continue
    
    try:
        loaded_models[model_key] = load_model(model_path, compile=False)
        print(f"✅ Đã load thành công model {model_key}")
    except Exception as e:
        print(f"❌ Lỗi khi load model {model_key}: {str(e)}")

if not loaded_models:
    raise ValueError("Không load được model nào, không thể tiếp tục")

if not os.path.exists(SCALER_PATH):
    raise FileNotFoundError(f"Không tìm thấy scaler tại {SCALER_PATH}")

scaler = joblib.load(SCALER_PATH)
print("✅ Đã load xong models và scaler")

# Tạo bộ xử lý đầu vào
input_processor = create_input_processor()
print("✅ Đã khởi tạo bộ xử lý đầu vào")

# ==== FLASK APP ====
app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
last_prediction = {
    "timestamp": None,
    "total_records": 0,
    "models_results": {},
    "ensemble_result": {
        "attack_percentage": 0.0,
        "attack_count": 0,
        "normal_count": 0,
        "attack_types": {}
    },
    "preprocessing_info": {}
}

def detect_attack_with_model(X_input, model_key):
    """Phát hiện tấn công với một model cụ thể"""
    model = loaded_models[model_key]
    X_scaled = scaler.transform(X_input)
    
    # Xử lý đặc biệt cho từng loại model
    if model_key == "lstm":
        # LSTM cần dữ liệu dạng sequence (timestep, features)
        # Kiểm tra số lượng features
        expected_features = model.input_shape[-1]
        if X_scaled.shape[1] != expected_features and X_scaled.shape[1] > expected_features:
            print(f"⚠️ Số features ({X_scaled.shape[1]}) khác với yêu cầu của LSTM ({expected_features}), sẽ điều chỉnh")
            X_scaled = X_scaled[:, :expected_features]  # Chỉ lấy số lượng features cần thiết
        
        X_scaled = np.expand_dims(X_scaled, axis=1)
    else:
        # CNN cần dữ liệu dạng ảnh (samples, height, width, channels)
        # Kiểm tra số lượng features
        expected_features = model.input_shape[1:-1]  # Bỏ qua batch_size và channels
        total_expected = np.prod(expected_features)
        
        if X_scaled.shape[1] != total_expected and X_scaled.shape[1] > total_expected:
            print(f"⚠️ Số features ({X_scaled.shape[1]}) khác với yêu cầu của CNN ({total_expected}), sẽ điều chỉnh")
            X_scaled = X_scaled[:, :total_expected]  # Chỉ lấy số lượng features cần thiết
        
        # Thêm chiều channel
        X_scaled = np.expand_dims(X_scaled, axis=-1)
    
    # Dự đoán
    preds = model.predict(X_scaled)
    predicted_labels = np.argmax(preds, axis=1)
    confidences = np.max(preds, axis=1)
    
    return predicted_labels, confidences

@app.route("/")
def home():
    return render_template_string("""
    <html>
    <head>
        <title>IDS Multi-model Analysis</title>
        <style>
            body { font-family: Arial; padding: 20px; max-width: 900px; margin: 0 auto; }
            .result-box { background-color: #f5f5f5; padding: 15px; border-radius: 5px; margin-top: 20px; }
            .attack-high { color: #d32f2f; font-weight: bold; }
            .attack-medium { color: #f57c00; font-weight: bold; }
            .attack-low { color: #388e3c; font-weight: bold; }
            table { width: 100%; border-collapse: collapse; margin-top: 10px; }
            table, th, td { border: 1px solid #ddd; }
            th, td { padding: 8px; text-align: left; }
            th { background-color: #f2f2f2; }
            .model-section { border-left: 4px solid #2196F3; padding-left: 10px; margin: 15px 0; }
            .model-title { color: #2196F3; margin-bottom: 10px; }
            .ensemble-section { border-left: 4px solid #4CAF50; padding-left: 10px; margin: 15px 0; }
            .ensemble-title { color: #4CAF50; margin-bottom: 10px; }
            .preprocessing-section { border-left: 4px solid #FF9800; padding-left: 10px; margin: 15px 0; }
            .preprocessing-title { color: #FF9800; margin-bottom: 10px; }
            .warning { color: #f44336; }
            .badge { display: inline-block; padding: 0.25em 0.6em; border-radius: 10rem; font-size: 75%; font-weight: 700; }
            .badge-info { background-color: #17a2b8; color: white; }
            .badge-warning { background-color: #ffc107; color: black; }
        </style>
    </head>
    <body>
        <h2>🛡️ IDS Multi-model Analysis</h2>

        <hr>
        <h3>📁 Upload file để phân tích</h3>
        <form action="/api/upload" method="post" enctype="multipart/form-data">
            <input type="file" name="file" required>
            <input type="submit" value="Phân tích">
        </form>
        <p><small>Hỗ trợ nhiều loại dữ liệu: CSV, log, PCAP, JSON, XML, Excel, ...</small></p>

        {% if timestamp %}
        <div class="result-box">
            <h3>Kết quả phân tích</h3>
            <p><strong>Thời gian:</strong> {{ timestamp }}</p>
            <p><strong>Tổng số bản ghi:</strong> {{ total_records }}</p>
            
            <!-- Kết quả của từng model -->
            {% for model_key, result in models_results.items() %}
            <div class="model-section">
                <h4 class="model-title">{{ result.display_name }}</h4>
                <p>
                    <strong>Tỉ lệ tấn công:</strong> 
                    <span class="{{ 'attack-high' if result.attack_percentage > 30 else 'attack-medium' if result.attack_percentage > 10 else 'attack-low' }}">
                        {{ result.attack_percentage|round(2) }}%
                    </span>
                </p>
                <p><strong>Số bản ghi thông thường:</strong> {{ result.normal_count }} ({{ (result.normal_count/total_records*100)|round(2) }}%)</p>
                <p><strong>Số bản ghi tấn công:</strong> {{ result.attack_count }} ({{ (result.attack_count/total_records*100)|round(2) }}%)</p>
                
                {% if result.attack_types %}
                <h5>Chi tiết loại tấn công:</h5>
                <table>
                    <tr>
                        <th>Loại tấn công</th>
                        <th>Số lượng</th>
                        <th>Tỉ lệ (%)</th>
                    </tr>
                    {% for attack_type, count in result.attack_types.items() %}
                    <tr>
                        <td>{{ attack_type }}</td>
                        <td>{{ count }}</td>
                        <td>{{ (count/total_records*100)|round(2) }}%</td>
                    </tr>
                    {% endfor %}
                </table>
                {% endif %}
            </div>
            {% endfor %}
            
            <!-- Kết quả Ensemble -->
            <div class="ensemble-section">
                <h4 class="ensemble-title">Kết quả Tổng hợp (Ensemble)</h4>
                <p>
                    <strong>Tỉ lệ tấn công:</strong> 
                    <span class="{{ 'attack-high' if ensemble_result.attack_percentage > 30 else 'attack-medium' if ensemble_result.attack_percentage > 10 else 'attack-low' }}">
                        {{ ensemble_result.attack_percentage|round(2) }}%
                    </span>
                </p>
                <p><strong>Số bản ghi thông thường:</strong> {{ ensemble_result.normal_count }} ({{ (ensemble_result.normal_count/total_records*100)|round(2) }}%)</p>
                <p><strong>Số bản ghi tấn công:</strong> {{ ensemble_result.attack_count }} ({{ (ensemble_result.attack_count/total_records*100)|round(2) }}%)</p>
                
                {% if ensemble_result.attack_types %}
                <h5>Chi tiết loại tấn công:</h5>
                <table>
                    <tr>
                        <th>Loại tấn công</th>
                        <th>Số lượng</th>
                        <th>Tỉ lệ (%)</th>
                    </tr>
                    {% for attack_type, count in ensemble_result.attack_types.items() %}
                    <tr>
                        <td>{{ attack_type }}</td>
                        <td>{{ count }}</td>
                        <td>{{ (count/total_records*100)|round(2) }}%</td>
                    </tr>
                    {% endfor %}
                </table>
                {% endif %}
            </div>
        </div>
        {% endif %}
    </body>
    </html>
    """, **last_prediction)

@app.route("/api/predict", methods=["GET"])
def api_predict():
    # Tạm thời tắt chức năng realtime
    return jsonify({"status": "error", "message": "Chức năng realtime tạm thời không khả dụng."})

@app.route("/api/upload", methods=["POST"])
def upload_and_predict():
    global last_prediction
    if "file" not in request.files:
        return jsonify({"status": "error", "message": "Không tìm thấy file trong yêu cầu."})
    
    file = request.files["file"]
    if file.filename == "":
        return jsonify({"status": "error", "message": "File không hợp lệ."})
    
    try:
        filepath = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
        file.save(filepath)
        print(f"✅ Đã lưu file tại: {filepath}")

        # Sử dụng bộ tiền xử lý để xử lý file
        print(f"🔄 Đang tiền xử lý file {file.filename}...")
        try:
            df, preprocessing_info = input_processor.process_file(filepath)
            print(f"✅ Tiền xử lý hoàn tất: {len(df)} bản ghi với {df.shape[1]} features")
        except Exception as e:
            print(f"❌ Lỗi khi tiền xử lý: {str(e)}")
            return jsonify({"status": "error", "message": f"Lỗi khi tiền xử lý: {str(e)}"})
        
        # Kết quả của từng model
        models_results = {}
        all_predictions = {}  # Lưu tất cả dự đoán để ensemble
        
        # Phân tích với từng model
        for model_key, model_info in MODELS.items():
            if model_key not in loaded_models:
                continue  # Bỏ qua models không load được
                
            print(f"🔍 Đang phân tích với model {model_key}...")
            
            try:
                # Xử lý theo batch nếu dữ liệu lớn
                BATCH_SIZE = 1000
                all_labels = []
                all_confidences = []
                
                for i in range(0, len(df), BATCH_SIZE):
                    batch = df.iloc[i:i+BATCH_SIZE].values
                    batch_labels, batch_confidences = detect_attack_with_model(batch, model_key)
                    all_labels.extend(batch_labels)
                    all_confidences.extend(batch_confidences)
                
                # Lưu tất cả dự đoán cho ensemble
                all_predictions[model_key] = all_labels
                
                # Phân tích kết quả cho model hiện tại
                attack_count = sum(1 for label in all_labels if label != 0)
                normal_count = len(all_labels) - attack_count
                attack_percentage = (attack_count / len(all_labels)) * 100 if len(all_labels) > 0 else 0
                
                # Đếm số lượng từng loại tấn công
                attack_types_counter = Counter(all_labels)
                attack_types = {}
                
                # Sử dụng tên tấn công từ mapping
                for attack_id, count in attack_types_counter.items():
                    if attack_id != 0:  # Bỏ qua normal (0)
                        attack_name = ATTACK_TYPES.get(attack_id, f"Loại không xác định ({attack_id})")
                        attack_types[attack_name] = count
                
                # Sắp xếp theo số lượng giảm dần
                attack_types = dict(sorted(attack_types.items(), key=lambda x: x[1], reverse=True))
                
                # Lưu kết quả của model hiện tại
                models_results[model_key] = {
                    "display_name": model_info["display_name"],
                    "attack_percentage": attack_percentage,
                    "attack_count": attack_count,
                    "normal_count": normal_count,
                    "attack_types": attack_types
                }
            except Exception as e:
                print(f"❌ Lỗi khi phân tích với model {model_key}: {str(e)}")
                continue
        
        if not models_results:
            return jsonify({"status": "error", "message": "Không thể phân tích với bất kỳ model nào."})
        
        # Tổng hợp kết quả (ensemble) bằng phương pháp voting
        ensemble_predictions = []
        for i in range(len(df)):
            votes = []
            for model_key in all_predictions:
                if i < len(all_predictions[model_key]):
                    votes.append(all_predictions[model_key][i])
            
            if votes:
                # Lấy kết quả phổ biến nhất
                most_common = Counter(votes).most_common(1)[0][0]
                ensemble_predictions.append(most_common)
        
        # Tính toán kết quả ensemble
        ensemble_attack_count = sum(1 for label in ensemble_predictions if label != 0)
        ensemble_normal_count = len(ensemble_predictions) - ensemble_attack_count
        ensemble_attack_percentage = (ensemble_attack_count / len(ensemble_predictions)) * 100 if ensemble_predictions else 0
        
        # Đếm số lượng từng loại tấn công cho ensemble
        ensemble_attack_types_counter = Counter(ensemble_predictions)
        ensemble_attack_types = {}
        
        for attack_id, count in ensemble_attack_types_counter.items():
            if attack_id != 0:
                attack_name = ATTACK_TYPES.get(attack_id, f"Loại không xác định ({attack_id})")
                ensemble_attack_types[attack_name] = count
        
        ensemble_attack_types = dict(sorted(ensemble_attack_types.items(), key=lambda x: x[1], reverse=True))
        
        # Cập nhật kết quả tổng hợp
        last_prediction = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "total_records": len(df),
            "models_results": models_results,
            "ensemble_result": {
                "attack_percentage": ensemble_attack_percentage,
                "attack_count": ensemble_attack_count,
                "normal_count": ensemble_normal_count,
                "attack_types": ensemble_attack_types
            },
            "preprocessing_info": preprocessing_info
        }
        
        print(f"✅ Đã hoàn thành phân tích với {len(models_results)} models")

        # Redirect về trang chủ để hiển thị kết quả
        return redirect(url_for("home"))

    except Exception as e:
        print(f"❌ Lỗi: {str(e)}")
        return jsonify({"status": "error", "message": str(e)})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
