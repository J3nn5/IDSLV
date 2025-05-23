import os
import time
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template_string, redirect, url_for
from keras.models import load_model
import joblib
from collections import Counter
from tienxuly import create_input_processor

#==============================================================================
# CẤU HÌNH HỆ THỐNG
#==============================================================================

# Định nghĩa các model sẽ sử dụng
MODELS = {
    "cnn": {
        "path": "models/cnn.h5",           # Đường dẫn đến file model
        "display_name": "CNN"              # Tên hiển thị
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

# Đường dẫn đến scaler
SCALER_PATH = "models/scaler.pkl"

# Thư mục lưu files tải lên
UPLOAD_FOLDER = "uploads"

# Mapping mã loại tấn công sang tên dễ đọc
ATTACK_TYPES = {
    0: "Normal",            # Không phải tấn công
    1: "DoS",               # Denial of Service
    2: "Probe",             # Scan/Probe
    3: "R2L",               # Remote to Local
    4: "U2R",               # User to Root
    5: "Backdoor",          # Backdoor attacks
    6: "Exploits",          # Các cuộc tấn công khai thác lỗ hổng
    7: "Analysis",          # Tấn công phân tích
    8: "Fuzzers",           # Tấn công fuzzing
    9: "Worms",             # Virus worm
    10: "Shellcode",        # Tấn công shellcode
    11: "Generic"           # Tấn công chung
}

# Tạo thư mục uploads nếu chưa tồn tại
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

#==============================================================================
# TẢI MÔ HÌNH VÀ SCALER
#==============================================================================

print("🔄 Đang load models và scaler...")
loaded_models = {}

# Vòng lặp qua tất cả model đã định nghĩa
for model_key, model_info in MODELS.items():
    model_path = model_info["path"]
    
    # Kiểm tra file model có tồn tại không
    if not os.path.exists(model_path):
        print(f"⚠️ Không tìm thấy model {model_key} tại {model_path}, sẽ bỏ qua")
        continue
    
    try:
        # Thử load model bằng Keras
        loaded_models[model_key] = load_model(model_path, compile=False)
        print(f"✅ Đã load thành công model {model_key}")
    except Exception as e:
        # Xử lý lỗi khi load model
        print(f"❌ Lỗi khi load model {model_key}: {str(e)}")

# Kiểm tra xem có model nào được load thành công không
if not loaded_models:
    raise ValueError("Không load được model nào, không thể tiếp tục")

# Kiểm tra file scaler
if not os.path.exists(SCALER_PATH):
    raise FileNotFoundError(f"Không tìm thấy scaler tại {SCALER_PATH}")

# Load scaler
scaler = joblib.load(SCALER_PATH)
print("✅ Đã load xong models và scaler")

# Tạo bộ xử lý đầu vào
input_processor = create_input_processor()
print("✅ Đã khởi tạo bộ xử lý đầu vào")

#==============================================================================
# KHỞI TẠO FLASK
#==============================================================================

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# Biến lưu kết quả phân tích gần nhất
last_prediction = {
    "timestamp": None,                      # Thời gian phân tích
    "total_records": 0,                     # Tổng số bản ghi đã phân tích
    "models_results": {},                   # Kết quả từng model
    "ensemble_result": {                    # Kết quả tổng hợp
        "attack_percentage": 0.0,
        "attack_count": 0,
        "normal_count": 0,
        "attack_types": {}
    },
    "preprocessing_info": {}                # Thông tin tiền xử lý
}

#==============================================================================
# CÁC HÀM CHỨC NĂNG
#==============================================================================

def format_analysis_time(seconds):
    """
    Định dạng thời gian phân tích thành dạng dễ đọc
    
    Parameters:
    -----------
    seconds : float
        Thời gian tính bằng giây
        
    Returns:
    --------
    str
        Chuỗi thời gian định dạng (ví dụ: "1d 2h 31' 45''")
    """
    if seconds < 0:
        return "0''"
    
    # Tính toán các đơn vị thời gian
    days = int(seconds // 86400)        # 1 ngày = 86400 giây
    hours = int((seconds % 86400) // 3600)  # 1 giờ = 3600 giây
    minutes = int((seconds % 3600) // 60)   # 1 phút = 60 giây
    secs = int(seconds % 60)                # Giây còn lại
    
    # Tạo chuỗi kết quả
    parts = []
    
    if days > 0:
        parts.append(f"{days}d")
    if hours > 0:
        parts.append(f"{hours}h")
    if minutes > 0:
        parts.append(f"{minutes}'")
    if secs > 0 or len(parts) == 0:  # Luôn hiển thị giây nếu không có đơn vị nào khác
        parts.append(f"{secs}''")
    
    return " ".join(parts)

def detect_attack_with_model(X_input, model_key):
    """
    Phát hiện tấn công với một model cụ thể
    
    Parameters:
    -----------
    X_input : numpy.ndarray
        Dữ liệu đầu vào cần phân tích
    model_key : str
        Tên model sẽ dùng để phân tích
        
    Returns:
    --------
    predicted_labels : numpy.ndarray
        Nhãn dự đoán cho mỗi mẫu đầu vào
    confidences : numpy.ndarray
        Độ tin cậy của các dự đoán
    """
    model = loaded_models[model_key]
    # Scale dữ liệu đầu vào để phù hợp với phân phối của tập huấn luyện
    X_scaled = scaler.transform(X_input)
    
    #------------------------------------------------------------------------------
    # Xử lý đặc biệt cho từng loại model
    #------------------------------------------------------------------------------
    if model_key == "lstm":
        # LSTM cần dữ liệu dạng sequence (timestep, features)
        # Kiểm tra số lượng features
        expected_features = model.input_shape[-1]
        
        # Điều chỉnh số lượng features nếu cần
        if X_scaled.shape[1] != expected_features:
            if X_scaled.shape[1] > expected_features:
                print(f"⚠️ Số features ({X_scaled.shape[1]}) nhiều hơn yêu cầu của LSTM ({expected_features}), sẽ cắt bớt")
                X_scaled = X_scaled[:, :expected_features]  # Chỉ lấy số lượng features cần thiết
            else:
                # Trường hợp dữ liệu có ít features hơn cần thiết
                print(f"⚠️ Số features ({X_scaled.shape[1]}) ít hơn yêu cầu của LSTM ({expected_features}), sẽ thêm features giả")
                padding = np.zeros((X_scaled.shape[0], expected_features - X_scaled.shape[1]))
                X_scaled = np.hstack([X_scaled, padding])
        
        # Thêm chiều timestep cho LSTM
        X_scaled = np.expand_dims(X_scaled, axis=1)
        
    elif model_key == "cnn":
        # CNN cần dữ liệu dạng ảnh (samples, height, width, channels)
        # Lấy thông tin input shape từ model
        input_shape = model.input_shape
        
        # Đảm bảo đây là mô hình CNN
        if len(input_shape) == 4:
            expected_features = input_shape[1] * input_shape[2]  # height * width
            
            # Điều chỉnh số lượng features nếu cần
            if X_scaled.shape[1] != expected_features:
                if X_scaled.shape[1] > expected_features:
                    print(f"⚠️ Số features ({X_scaled.shape[1]}) nhiều hơn yêu cầu của CNN ({expected_features}), sẽ cắt bớt")
                    X_scaled = X_scaled[:, :expected_features]
                else:
                    print(f"⚠️ Số features ({X_scaled.shape[1]}) ít hơn yêu cầu của CNN ({expected_features}), sẽ thêm features giả")
                    padding = np.zeros((X_scaled.shape[0], expected_features - X_scaled.shape[1]))
                    X_scaled = np.hstack([X_scaled, padding])
            
            # Reshape thành dạng ảnh 2D + kênh
            X_scaled = X_scaled.reshape(-1, input_shape[1], input_shape[2], 1)
        else:
            # Nếu không xác định được shape, sử dụng phương pháp mặc định
            X_scaled = np.expand_dims(X_scaled, axis=-1)
            
    elif model_key == "logistic":
        # Logistic Regression có thể là model Keras hoặc model scikit-learn
        # Kiểm tra loại model và xử lý phù hợp
        if hasattr(model, 'input_shape'):  # Keras model
            expected_features = model.input_shape[1]
            
            # Điều chỉnh số lượng features nếu cần
            if X_scaled.shape[1] != expected_features:
                if X_scaled.shape[1] > expected_features:
                    print(f"⚠️ Số features ({X_scaled.shape[1]}) nhiều hơn yêu cầu của Logistic ({expected_features}), sẽ cắt bớt")
                    X_scaled = X_scaled[:, :expected_features]
                else:
                    print(f"⚠️ Số features ({X_scaled.shape[1]}) ít hơn yêu cầu của Logistic ({expected_features}), sẽ thêm features giả")
                    padding = np.zeros((X_scaled.shape[0], expected_features - X_scaled.shape[1]))
                    X_scaled = np.hstack([X_scaled, padding])
        else:
            # Giả sử đây là scikit-learn model, không cần xử lý đặc biệt
            pass
    
    #------------------------------------------------------------------------------
    # Dự đoán dựa trên loại model
    #------------------------------------------------------------------------------
    if model_key == "logistic" and not hasattr(model, 'predict_proba'):
        # Đây là model Keras Logistic
        preds = model.predict(X_scaled)
        predicted_labels = np.argmax(preds, axis=1)
        confidences = np.max(preds, axis=1)
    elif model_key == "logistic":
        # Đây là scikit-learn Logistic Regression model
        predicted_labels = model.predict(X_scaled)
        try:
            # Cố gắng lấy confidence scores nếu có
            probs = model.predict_proba(X_scaled)
            confidences = np.array([probs[i, label] for i, label in enumerate(predicted_labels)])
        except:
            # Nếu không có predict_proba, gán confidence = 1
            confidences = np.ones(len(predicted_labels))
    else:
        # Dự đoán cho các model khác (CNN, LSTM)
        preds = model.predict(X_scaled)
        predicted_labels = np.argmax(preds, axis=1)
        confidences = np.max(preds, axis=1)
    
    return predicted_labels, confidences

#==============================================================================
# ROUTES FLASK
#==============================================================================

@app.route("/")
def home():
    """Trang chủ - Hiển thị form upload và kết quả phân tích nếu có"""
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
            .dataset-name { font-weight: bold; font-size: 1.1em; color: #333; background-color: #e3f2fd; padding: 5px 10px; border-radius: 4px; display: inline-block; margin-bottom: 10px; }
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
            <p><strong>Tập dữ liệu:</strong> <span class="dataset-name">{{ dataset_name }}</span></p>
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
                <p><strong>⏱️ Thời gian phân tích:</strong> {{ result.analysis_time }}</p>
                
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
            
        </div>
        {% endif %}
    </body>
    </html>
    """, **last_prediction)

@app.route("/api/predict", methods=["GET"])
def api_predict():
    """API endpoint cho dự đoán realtime (hiện tại đã tắt)"""
    # Tạm thời tắt chức năng realtime
    return jsonify({"status": "error", "message": "Chức năng realtime tạm thời không khả dụng."})

@app.route("/api/upload", methods=["POST"])
def upload_and_predict():
    """
    API endpoint xử lý file upload và thực hiện phân tích
    
    Quy trình:
    1. Nhận file từ client
    2. Lưu file vào thư mục upload
    3. Xử lý file với IDSPreprocessor
    4. Phân tích với từng model
    5. Tổng hợp kết quả và hiển thị
    """
    global last_prediction
    
    # Kiểm tra có file trong request không
    if "file" not in request.files:
        return jsonify({"status": "error", "message": "Không tìm thấy file trong yêu cầu."})
    
    file = request.files["file"]
    if file.filename == "":
        return jsonify({"status": "error", "message": "File không hợp lệ."})
    
    try:
        # Lưu file upload vào thư mục tạm
        filepath = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
        file.save(filepath)
        print(f"✅ Đã lưu file tại: {filepath}")

        #----------------------------------------------------------------------
        # BƯỚC 1: TIỀN XỬ LÝ FILE
        #----------------------------------------------------------------------
        print(f"🔄 Đang tiền xử lý file {file.filename}...")
        try:
            # Gọi processor để xử lý file đầu vào
            df, preprocessing_info = input_processor.process_file(filepath)
            print(f"✅ Tiền xử lý hoàn tất: {len(df)} bản ghi với {df.shape[1]} features")
        except Exception as e:
            print(f"❌ Lỗi khi tiền xử lý: {str(e)}")
            return jsonify({"status": "error", "message": f"Lỗi khi tiền xử lý: {str(e)}"})
        
        #----------------------------------------------------------------------
        # BƯỚC 2: PHÂN TÍCH VỚI TỪNG MODEL
        #----------------------------------------------------------------------
        models_results = {}                  # Lưu kết quả từng model
        all_predictions = {}                 # Lưu dự đoán để ensemble
        
        # Phân tích với từng model
        for model_key, model_info in MODELS.items():
            # Bỏ qua models không load được
            if model_key not in loaded_models:
                continue
                
            print(f"🔍 Đang phân tích với model {model_key}...")
            
            try:
                # Đo thời gian bắt đầu
                model_start_time = time.time()

                # Xử lý theo batch nếu dữ liệu lớn
                BATCH_SIZE = 1000
                all_labels = []              # Lưu nhãn dự đoán
                all_confidences = []         # Lưu độ tin cậy
                
                # Xử lý từng batch để tránh quá tải bộ nhớ
                for i in range(0, len(df), BATCH_SIZE):
                    batch = df.iloc[i:i+BATCH_SIZE].values
                    batch_labels, batch_confidences = detect_attack_with_model(batch, model_key)
                    all_labels.extend(batch_labels)
                    all_confidences.extend(batch_confidences)
                
                # Đo thời gian kết thúc
                model_end_time = time.time()
                analysis_time_seconds = model_end_time - model_start_time
                analysis_time_formatted = format_analysis_time(analysis_time_seconds)

                # Lưu tất cả dự đoán cho ensemble
                all_predictions[model_key] = all_labels
                
                #--------------------------------------------------------------
                # BƯỚC 3: PHÂN TÍCH KẾT QUẢ CHO MODEL
                #--------------------------------------------------------------
                # Đếm số lượng mẫu tấn công và mẫu bình thường
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
                    "attack_types": attack_types,
                    "analysis_time": analysis_time_formatted,    # Thời gian định dạng
                    "analysis_time_raw": analysis_time_seconds   # Thời gian thô (giây) để sử dụng trong tính toán khác
                }
                
                print(f"✅ Hoàn thành phân tích với {model_key} trong {analysis_time_formatted}")
                
            except Exception as e:
                # Ghi nhận lỗi và bỏ qua model này
                print(f"❌ Lỗi khi phân tích với model {model_key}: {str(e)}")
                continue

        # Kiểm tra có model nào cho kết quả không
        if not models_results:
            return jsonify({"status": "error", "message": "Không thể phân tích với bất kỳ model nào."})
        
        #----------------------------------------------------------------------
        # BƯỚC 4: TẠO KẾT QUẢ ENSEMBLE BẰNG PHƯƠNG PHÁP VOTING
        #----------------------------------------------------------------------
        ensemble_predictions = []
        
        # Voting cho từng mẫu
        for i in range(len(df)):
            votes = []
            for model_key in all_predictions:
                if i < len(all_predictions[model_key]):
                    votes.append(all_predictions[model_key][i])
            
            if votes:
                # Lấy kết quả phổ biến nhất (majority voting)
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
            if attack_id != 0:  # Bỏ qua normal (0)
                attack_name = ATTACK_TYPES.get(attack_id, f"Loại không xác định ({attack_id})")
                ensemble_attack_types[attack_name] = count
        
        # Sắp xếp theo số lượng giảm dần
        ensemble_attack_types = dict(sorted(ensemble_attack_types.items(), key=lambda x: x[1], reverse=True))
        
        #----------------------------------------------------------------------
        # BƯỚC 5: CHUẨN BỊ KẾT QUẢ CUỐI CÙNG
        #----------------------------------------------------------------------
        last_prediction = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),  # Thời gian phân tích
            "total_records": len(df),                         # Tổng số bản ghi
            "models_results": models_results,                 # Kết quả từng model
            "ensemble_result": {                              # Kết quả ensemble
                "attack_percentage": ensemble_attack_percentage,
                "attack_count": ensemble_attack_count,
                "normal_count": ensemble_normal_count,
                "attack_types": ensemble_attack_types
            },
            "preprocessing_info": preprocessing_info,         # Thông tin tiền xử lý
            "dataset_name": file.filename                     # Tên tập dữ liệu
        }
        
        print(f"✅ Đã hoàn thành phân tích với {len(models_results)} models")

        # Redirect về trang chủ để hiển thị kết quả
        return redirect(url_for("home"))

    except Exception as e:
        # Xử lý ngoại lệ không lường trước
        print(f"❌ Lỗi: {str(e)}")
        return jsonify({"status": "error", "message": str(e)})

#==============================================================================
# CHẠY ỨNG DỤNG
#==============================================================================

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
