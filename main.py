import json
import time
from pathlib import Path
from fastapi.responses import JSONResponse
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
from PIL import Image
from io import BytesIO
import torch
from transformers import AutoProcessor, AutoModelForImageTextToText

# ========== Configuración FastAPI ==========
app = FastAPI()
origins = ["http://localhost:3000", "http://127.0.0.1:3000"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ========== Modelo de datos para texto ==========
class TextoEntrada(BaseModel):
    texto: str

# ========== Cargar modelo y procesador ==========
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Usando dispositivo: {device}")

print("Cargando modelo y procesador MedGemma...")
model_id = "google/medgemma-4b-it"
processor = AutoProcessor.from_pretrained(model_id)
model = AutoModelForImageTextToText.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
    device_map="auto"
)
model.to(device)
print("Modelo cargado correctamente.")

# ========== Endpoint para TEXTO ==========
@app.post("/medgemma")
async def medgemma_texto(entrada: TextoEntrada):
    tiempo_inicio = time.time()
    prompt = entrada.texto  # SOLO usa el prompt recibido, NO añadas nada extra

    messages = [
        {"role": "user", "content": [{"type": "text", "text": prompt}]}
    ]
    inputs = processor.apply_chat_template(
        messages,
        return_tensors="pt",
        tokenize=True
    ).to(device)

    with torch.inference_mode():
        outputs = model.generate(input_ids=inputs, max_new_tokens=280, do_sample=False)

    decoded = processor.decode(outputs[0], skip_special_tokens=True)
    tiempo_fin = time.time()
    print(f"[TIEMPO] /medgemma: {round(tiempo_fin - tiempo_inicio, 2)} segundos")
    return {"respuesta": decoded}

# ========== Endpoint para IMAGEN ==========
@app.post("/medgemma-img")
async def medgemma_imagen(
    files: List[UploadFile] = File(...),
    prompt: Optional[str] = Form("Analiza esta imagen médica y proporciona un informe detallado en español.")
):
    tiempo_inicio = time.time()
    results = []
    for file in files:
        img_bytes = await file.read()
        image = Image.open(BytesIO(img_bytes)).convert("RGB")

        # Usa el prompt recibido en el form-data (frontend/cliente debe mandarlo)
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image", "image": image}
                ]
            }
        ]
        inputs = processor.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True,
            return_dict=True, return_tensors="pt"
        ).to(model.device, dtype=torch.bfloat16 if device == "cuda" else torch.float32)

        input_len = inputs["input_ids"].shape[-1]

        with torch.inference_mode():
            outputs = model.generate(**inputs, max_new_tokens=350, do_sample=False)

        generated = outputs[0][input_len:]
        decoded = processor.decode(generated, skip_special_tokens=True)

        results.append({
            "filename": file.filename,
            "descripcion": decoded
        })

    tiempo_fin = time.time()
    print(f"[TIEMPO] /medgemma-img: {round(tiempo_fin - tiempo_inicio, 2)} segundos")
    return {"resultados": results}
