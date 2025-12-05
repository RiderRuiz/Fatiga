import base64
import json
import os

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from kafka import KafkaProducer

from core.fatigue_engine import FatigueEngine

EYE_MODEL_PATH = "runs/detect/eye_yolo/weights/best.pt"
YAWN_MODEL_PATH = "runs/detect/yawn_yolo/weights/best.pt"
LOG_PATH = "data/fatigue_data.csv"
KAFKA_BOOTSTRAP = os.getenv("KAFKA_BOOTSTRAP", "localhost:9092")
KAFKA_TOPIC = os.getenv("KAFKA_TOPIC", "fatigue-events")

os.makedirs("data", exist_ok=True)
engine = FatigueEngine(EYE_MODEL_PATH, YAWN_MODEL_PATH, log_path=LOG_PATH)
engine.start_calibration()

producer: KafkaProducer | None = None
try:
    producer = KafkaProducer(
        bootstrap_servers=KAFKA_BOOTSTRAP,
        value_serializer=lambda v: json.dumps(v).encode("utf-8"),
        linger_ms=50,
        retries=3,
    )
except Exception:
    producer = None

app = FastAPI(title="Fatigue Detector API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", response_class=HTMLResponse)
async def index():
    return """
    <!DOCTYPE html>
    <html lang="es">
    <head>
      <meta charset="UTF-8" />
      <meta name="viewport" content="width=device-width, initial-scale=1" />
      <title>Detector de Fatiga - Web</title>
      <style>
        body {
          font-family: Arial, sans-serif;
          margin: 0;
          padding: 0;
          background: #0f172a;
          color: #e5e7eb;
          display: flex;
          justify-content: center;
        }
        .container {
          max-width: 900px;
          width: 100%;
          padding: 16px;
        }
        h1 {
          margin-top: 0;
          font-size: 1.5rem;
          text-align: center;
        }
        p {
          text-align: center;
          font-size: 0.9rem;
          color: #9ca3af;
        }
        .layout {
          display: flex;
          flex-wrap: wrap;
          gap: 16px;
          margin-top: 16px;
        }
        .panel {
          background: #111827;
          border-radius: 8px;
          padding: 12px;
          flex: 1 1 260px;
        }
        .panel h2 {
          font-size: 1rem;
          margin-top: 0;
          margin-bottom: 8px;
        }
        video {
          width: 100%;
          border-radius: 6px;
          border: 1px solid #374151;
        }
        #annotated {
          width: 100%;
          border-radius: 6px;
          border: 1px solid #374151;
        }
        #status {
          margin-top: 8px;
          font-weight: bold;
        }
        #metrics {
          margin-top: 8px;
          white-space: pre-wrap;
          font-family: "SF Mono", ui-monospace, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace;
          font-size: 0.85rem;
        }
        button {
          padding: 6px 12px;
          margin-right: 8px;
          border-radius: 9999px;
          border: none;
          cursor: pointer;
          font-size: 0.9rem;
        }
        #startBtn {
          background-color: #22c55e;
          color: #022c22;
        }
        #startBtn:disabled {
          background-color: #15803d;
          cursor: not-allowed;
        }
        #stopBtn {
          background-color: #f97316;
          color: #111827;
        }
        #stopBtn:disabled {
          background-color: #7c2d12;
          cursor: not-allowed;
        }
        @media (max-width: 640px) {
          .layout {
            flex-direction: column;
          }
          h1 {
            font-size: 1.25rem;
          }
        }
      </style>
    </head>
    <body>
      <div class="container">
        <h1>Detector de Fatiga (YOLO) - Web</h1>
        <p>Permite acceso a la cámara, la imagen se procesa en el servidor y ves el resultado anotado en tiempo casi real.</p>
        <div class="layout">
          <div class="panel">
            <h2>Vista previa de la cámara</h2>
            <video id="video" autoplay playsinline muted></video>
            <canvas id="canvas" width="640" height="480" style="display:none;"></canvas>
            <div style="margin-top:8px;">
              <button id="startBtn">Iniciar detección</button>
              <button id="stopBtn" disabled>Detener</button>
            </div>
            <div id="status">Estado: parado</div>
          </div>
          <div class="panel">
            <h2>Salida del modelo</h2>
            <img id="annotated" alt="Frame anotado" />
            <div id="metrics"></div>
          </div>
        </div>
      </div>

      <script>
        const video = document.getElementById('video');
        const canvas = document.getElementById('canvas');
        const ctx = canvas.getContext('2d');
        const startBtn = document.getElementById('startBtn');
        const stopBtn = document.getElementById('stopBtn');
        const statusDiv = document.getElementById('status');
        const annotatedImg = document.getElementById('annotated');
        const metricsDiv = document.getElementById('metrics');

        let streaming = false;
        let intervalId = null;

        async function startCamera() {
          try {
            const stream = await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
            video.srcObject = stream;
            await video.play();
            return true;
          } catch (err) {
            console.error('Error accediendo a la cámara:', err);
            statusDiv.textContent = 'Error accediendo a la cámara: ' + err;
            return false;
          }
        }

        async function sendFrame() {
          if (!streaming) return;
          const w = video.videoWidth || 640;
          const h = video.videoHeight || 480;
          canvas.width = w;
          canvas.height = h;
          ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

          canvas.toBlob(async (blob) => {
            if (!blob) return;
            const formData = new FormData();
            formData.append('frame', blob, 'frame.jpg');
            try {
              const res = await fetch('/infer', {
                method: 'POST',
                body: formData,
              });
              if (!res.ok) {
                const errText = await res.text();
                statusDiv.textContent = 'Error del backend: ' + errText;
                return;
              }
              const data = await res.json();
              statusDiv.textContent = 'Estado: ' + data.estado;
              metricsDiv.textContent =
                'PERCLOS: ' + data.perclos.toFixed(1) + '%\\n' +
                'PERCLOS medio: ' + data.avg_perclos.toFixed(1) + '%\\n' +
                'Yawn ratio: ' + data.yawn_ratio.toFixed(2) + '\\n' +
                'FPS: ' + data.fps + '\\n' +
                'Rostro presente: ' + (data.face_present ? 'Sí' : 'No');
              if (data.annotated_frame) {
                annotatedImg.src = 'data:image/jpeg;base64,' + data.annotated_frame;
              }
            } catch (err) {
              console.error('Error enviando frame:', err);
              statusDiv.textContent = 'Error enviando frame: ' + err;
            }
          }, 'image/jpeg', 0.8);
        }

        startBtn.addEventListener('click', async () => {
          if (streaming) return;
          const ok = await startCamera();
          if (!ok) return;
          streaming = true;
          startBtn.disabled = true;
          stopBtn.disabled = false;
          statusDiv.textContent = 'Estado: capturando y enviando frames...';
          intervalId = setInterval(sendFrame, 500); // cada 500 ms
        });

        stopBtn.addEventListener('click', () => {
          streaming = false;
          startBtn.disabled = false;
          stopBtn.disabled = true;
          statusDiv.textContent = 'Estado: parado';
          if (intervalId) {
            clearInterval(intervalId);
            intervalId = null;
          }
          const stream = video.srcObject;
          if (stream) {
            stream.getTracks().forEach((t) => t.stop());
            video.srcObject = null;
          }
        });
      </script>
    </body>
    </html>
    """


@app.post("/infer")
async def infer(frame: UploadFile = File(...)):
    try:
        content = await frame.read()
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"No se pudo leer el archivo: {exc}") from exc

    if not content:
        raise HTTPException(status_code=400, detail="El archivo esta vacio")

    image_array = np.frombuffer(content, dtype=np.uint8)
    frame_bgr = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
    if frame_bgr is None:
        raise HTTPException(status_code=400, detail="Imagen invalida")

    annotated, metrics = engine.step(frame_bgr)
    estado = metrics["estado"]

    # Registrar cuando no estamos calibrando ni sin rostro,
    # como en el flujo facialmesh original.
    if estado not in {"Calibrando", "Sin rostro"}:
        engine.log_metrics(metrics)
        if producer is not None:
            event = {
                "estado": estado,
                "perclos": float(metrics["perclos"]),
                "avg_perclos": float(metrics["avg_perclos"]),
                "yawn_ratio": float(metrics["yawn_ratio"]),
                "fps": int(metrics["fps"]),
                "face_present": bool(metrics.get("face_present", False)),
            }
            try:
                producer.send(KAFKA_TOPIC, event)
            except Exception:
                pass

    _, buffer = cv2.imencode(".jpg", annotated)
    annotated_base64 = base64.b64encode(buffer).decode("utf-8")

    payload = {
        "estado": estado,
        "perclos": metrics["perclos"],
        "avg_perclos": metrics["avg_perclos"],
        "yawn_ratio": metrics["yawn_ratio"],
        "fps": metrics["fps"],
        "face_present": metrics["face_present"],
        "annotated_frame": annotated_base64,
    }
    return JSONResponse(payload)


@app.post("/reset")
async def reset_engine():
    engine.reset()
    engine.start_calibration()
    return {"status": "ok"}
