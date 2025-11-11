# Bước 1: Bắt đầu từ Python 3.11 (phiên bản đã hoạt động)
FROM python:3.11-slim

# Đặt thư mục làm việc bên trong container
WORKDIR /code

# Bước 2: Cài đặt các thư viện hệ thống
# Cài đặt build-essential (chứa C++ compiler) và cmake
# Đây là bước quan trọng để 'kenlm' có thể build
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    libopencv-dev \
    && rm -rf /var/lib/apt/lists/*

# Bước 3: Cài đặt thư viện Python
# Copy file requirements trước để tận dụng cache
COPY requirements.txt requirements.txt
# BƯỚC 3.5: CÀI PYTORCH CHO LINUX (chính xác phiên bản cu126)
RUN pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Cài đặt CÁC THƯ VIỆN CÒN LẠI (đã xóa torch khỏi requirements.txt)
RUN pip install --no-cache-dir -r requirements.txt

# Bước 4: Copy toàn bộ code dự án
# Copy thư mục app (main.py, pipeline.py, static/index.html)
COPY ./app /code/app
# Copy thư mục models (3 file model của bạn)
COPY ./models /code/models

# Bước 5: Chạy ứng dụng
# Mở cổng 8000 cho web
EXPOSE 8000

# Lệnh để khởi động server FastAPI
# Nó sẽ chạy file 'main.py' bên trong thư mục 'app'
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]