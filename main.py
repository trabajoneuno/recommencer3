# main.py
from fastapi import FastAPI, HTTPException
from modelo_recomendacion import obtener_recomendaciones
from typing import Dict

app = FastAPI(title="API de Recomendaciones",
             description="API para recomendar productos similares",
             version="1.0.0")

@app.get("/")
def read_root():
    return {"message": "Bienvenido a la API de Recomendaciones"}

@app.get("/recomendar/{producto}")
def recomendar(producto: str) -> Dict[str, float]:
    recomendaciones = obtener_recomendaciones(producto)
    if not recomendaciones:
        raise HTTPException(status_code=404, detail="Producto no encontrado o error en las recomendaciones")
    return recomendaciones
