# Backend Service Dockerfile
FROM python:3.11-slim

# Install system dependencies required by OpenCV
RUN apt-get update && apt-get install -y libgl1 libglib2.0-0 && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code and models
COPY backend/ ./backend/
COPY core/ ./core/
COPY runs/detect/eye_yolo/weights/best.pt ./runs/detect/eye_yolo/weights/best.pt
COPY runs/detect/yawn_yolo/weights/best.pt ./runs/detect/yawn_yolo/weights/best.pt

# Expose the port the app runs on
EXPOSE 8000

# Run the application
CMD ["uvicorn", "backend.server:app", "--host", "0.0.0.0", "--port", "8000"]
