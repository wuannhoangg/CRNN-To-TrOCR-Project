import uvicorn
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import HTMLResponse
from starlette.staticfiles import StaticFiles
import os
import shutil
import argparse
import traceback

# Import hàm pipeline V4 của bạn
from .pipeline import run_pipeline

# Khởi tạo app FastAPI
app = FastAPI(title="Wuann's Transformer OCR API")

# "Mount" thư mục static để phục vụ file index.html
app.mount("/static", StaticFiles(directory="app/static"), name="static")

# Định nghĩa các đường dẫn model (để code sạch hơn)
# Các đường dẫn này là *bên trong* Docker container
DETECTION_MODEL_PATH = "models/DB_TD500_resnet50.onnx"
RECOGNITION_MODEL_PATH = "models/best_transformer.pt"
LM_MODEL_PATH = "models/3-gram-lm.binary"


@app.get("/", response_class=HTMLResponse)
async def get_frontend():
    """Phục vụ file frontend index.html"""
    with open("app/static/index.html", "r", encoding="utf-8") as f:
        html_content = f.read()
    return HTMLResponse(content=html_content)


@app.post("/ocr/predict")
async def ocr_predict(image: UploadFile = File(...)):
    """
    Endpoint chính: Nhận ảnh, chạy pipeline, và trả về text
    """
    
    # 1. Tạo một file tạm thời để lưu ảnh upload
    temp_input_path = f"temp_{image.filename}"
    try:
        with open(temp_input_path, "wb") as buffer:
            shutil.copyfileobj(image.file, buffer)
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Không thể lưu file tạm: {e}")
    finally:
        image.file.close()

    # 2. Chuẩn bị các tham số (args) để truyền cho hàm pipeline
    #    Chúng ta dùng argparse.Namespace để "giả lập" các tham số dòng lệnh
    output_temp_file = f"temp_output_{image.filename}.txt"
    
    args = argparse.Namespace(
        # Các đường dẫn file
        input_image=temp_input_path,
        detection_model=DETECTION_MODEL_PATH,
        recognition_model=RECOGNITION_MODEL_PATH,
        lm_model=LM_MODEL_PATH,
        output_file=output_temp_file,
        
        # Các tham số tinh chỉnh tốt nhất của bạn
        beam_width=30,
        lm_alpha=0.5,
        lm_beta=0.2
    )

    # 3. Chạy pipeline hoàn chỉnh
    full_text_result = ""
    try:
        print(f"[INFO] Bắt đầu chạy pipeline cho ảnh: {image.filename}")
        
        run_pipeline(args)
        
        # 4. Đọc kết quả từ file output
        if os.path.exists(output_temp_file):
            with open(output_temp_file, 'r', encoding='utf-8') as f:
                full_text_result = f.read()
        else:
            print("[WARN] Pipeline chạy xong nhưng không tạo ra file output.")
            
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Lỗi pipeline: {e}")
    
    finally:
        # 5. Dọn dẹp file tạm
        if os.path.exists(temp_input_path):
            os.remove(temp_input_path)
        if os.path.exists(output_temp_file):
            os.remove(output_temp_file)

    # 6. Trả về kết quả
    print(f"[INFO] Pipeline hoàn tất. Trả về kết quả.")
    return {"result_text": full_text_result}


if __name__ == "__main__":
    # Lệnh này dùng để chạy test ở máy (local)
    uvicorn.run(app, host="0.0.0.0", port=8000)