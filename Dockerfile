FROM python:3.11-slim

RUN apt-get update && apt-get install -y libgl1 libglib2.0-0 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY backend/ ./backend/
COPY core/ ./core/
COPY runs/detect/eye_yolo/weights/best.pt ./runs/detect/eye_yolo/weights/best.pt
COPY runs/detect/yawn_yolo/weights/best.pt ./runs/detect/yawn_yolo/weights/best.pt

EXPOSE 8000

CMD ["uvicorn", "backend.server:app", "--host", "0.0.0.0", "--port", "8000"]
