# Búsqueda Semántica con IA - Casos Clínicos: API MedGemma en Google Cloud VM

Este proyecto provee una **API REST** para inferencia médica basada en imágenes y texto clínico, usando el modelo **MedGemma 4B (`google/medgemma-4b-it`)**.  
Está diseñada para desplegarse en una **VM de Google Cloud**, optimizada para consultas rápidas desde frontends, otros backends o integraciones externas.

---

## Características principales

- **`/medgemma`**: Endpoint POST para consultas de texto clínico.
- **`/medgemma-img`**: Endpoint POST para análisis y reporte automático de imágenes médicas (una o varias).

**Tecnologías clave:**  
- FastAPI  
- PyTorch (con soporte CUDA opcional)  
- Transformers (`AutoModelForImageTextToText`)  
- Hugging Face Hub

---

## Estructura de Endpoints

### `POST /medgemma`
- **Descripción**: Genera respuesta médica basada en texto usando el modelo MedGemma.
- **Entrada**:  
  ```json
  {
    "texto": "Paciente de 57 años con dolor abdominal y fiebre, antecedentes de..."
  }
### `POST /medgemma-img`
- **Descripción**: Devuelve un informe automático generado por IA a partir de una o varias imágenes médicas enviadas.
- **Entrada**:  
  - files: archivos de imagen (formato multipart/form-data).
  - prompt: (opcional) instrucciones clínicas específicas en español.
