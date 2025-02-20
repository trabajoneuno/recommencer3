# main.py
import pandas as pd
from fastapi import FastAPI, HTTPException
import pickle
import os

# IMPORTANTE: La clase debe estar definida EXACTAMENTE igual que cuando creaste el modelo
class ModeloRecomendacion:
    def __init__(self, similarity_df):
        self.similarity_df = similarity_df
    
    def recomendar_producto(self, producto):
        if producto not in self.similarity_df.columns:
            return {}
        similar_producto = self.similarity_df[producto].sort_values(ascending=False).head(6)
        similar_producto = similar_producto[1:]
        return dict(similar_producto)

# Crear la aplicación FastAPI
app = FastAPI(
    title="API de Recomendaciones",
    description="API para recomendar productos similares",
    version="1.0.0"
)

# Cargar el modelo usando pickle en lugar de joblib
try:
    with open('modelo_recomendacion.pkl', 'rb') as file:
        modelo = pickle.load(file)
    print("Modelo cargado exitosamente")
except Exception as e:
    print(f"Error al cargar el modelo: {str(e)}")
    modelo = None

@app.get("/")
def read_root():
    if modelo is None:
        return {"message": "API activa pero modelo no cargado"}
    return {"message": "API activa y modelo cargado correctamente"}

@app.get("/recomendar/{producto}")
async def recomendar(producto: str):
    if modelo is None:
        raise HTTPException(
            status_code=500,
            detail="Modelo no disponible"
        )
    
    try:
        recomendaciones = modelo.recomendar_producto(producto)
        if not recomendaciones:
            raise HTTPException(
                status_code=404,
                detail=f"Producto '{producto}' no encontrado"
            )
        return recomendaciones
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error al procesar la recomendación: {str(e)}"
        )

@app.get("/productos-disponibles")
async def productos_disponibles():
    if modelo is None:
        raise HTTPException(
            status_code=500,
            detail="Modelo no disponible"
        )
    return {
        "total_productos": len(modelo.similarity_df.columns),
        "ejemplos": list(modelo.similarity_df.columns[:5])
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=10000)
