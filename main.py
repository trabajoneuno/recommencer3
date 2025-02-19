import os
import gc
import psutil
import numpy as np
import pandas as pd
from typing import Dict, List, Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from sklearn.preprocessing import LabelEncoder
import logging
import tempfile
from pathlib import Path
import traceback
import asyncio
from contextlib import asynccontextmanager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Clase para mantener el estado de la aplicación
class AppState:
    def __init__(self):
        self.is_loaded = False
        self.recommender = None

app_state = AppState()

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("Application startup")
    yield
    # Shutdown
    logger.info("Application shutdown")
    if app_state.recommender:
        app_state.recommender.embedding_store.cleanup()

# Create FastAPI app with lifespan
app = FastAPI(lifespan=lifespan)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Las clases MemoryEfficientStore y RecommendationSystem permanecen igual...
[Previous code for these classes remains the same]

async def load_data():
    """Carga asíncrona de datos"""
    try:
        logger.info("Starting async data loading")
        app_state.recommender = RecommendationSystem()
        
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        csv_path = os.path.join(BASE_DIR, "productos.csv")
        
        # Procesamiento de datos en segundo plano
        app_state.recommender.process_data(csv_path)
        
        app_state.is_loaded = True
        logger.info("Data loading completed successfully")
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        logger.error(traceback.format_exc())
        app_state.is_loaded = False

@app.on_event("startup")
async def startup_event():
    # Iniciar la carga de datos en segundo plano
    asyncio.create_task(load_data())

@app.get("/")
async def root():
    return {
        "status": "ok",
        "data_loaded": app_state.is_loaded,
        "memory_usage_mb": psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)
    }

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "data_loaded": app_state.is_loaded,
        "memory_usage_mb": psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024),
        "num_products": len(app_state.recommender.id_to_name) if app_state.recommender else 0
    }

@app.get("/recomendaciones")
async def get_recommendations(product_name: str, top_n: int = 5):
    if not app_state.is_loaded:
        raise HTTPException(status_code=503, detail="Service is still loading data")
    
    return {
        "producto": product_name,
        "recomendaciones": app_state.recommender.get_recommendations(product_name, top_n)
    }

@app.get("/productos")
async def list_products(
    page: Optional[int] = 1,
    page_size: Optional[int] = 50,
    search: Optional[str] = None
):
    if not app_state.is_loaded:
        raise HTTPException(status_code=503, detail="Service is still loading data")
    
    if page < 1:
        raise HTTPException(status_code=400, detail="Page number must be >= 1")
    if page_size < 1 or page_size > 100:
        raise HTTPException(status_code=400, detail="Page size must be between 1 and 100")
    
    products = list(app_state.recommender.name_to_id.keys())
    
    if search:
        search = search.lower()
        products = [p for p in products if search in p.lower()]
    
    products.sort()
    
    total_products = len(products)
    total_pages = (total_products + page_size - 1) // page_size
    
    if page > total_pages and total_pages > 0:
        raise HTTPException(status_code=404, detail="Page not found")
    
    start_idx = (page - 1) * page_size
    end_idx = min(start_idx + page_size, total_products)
    page_products = products[start_idx:end_idx]
    
    return {
        "productos": page_products,
        "pagina_actual": page,
        "total_paginas": total_pages,
        "productos_por_pagina": page_size,
        "total_productos": total_products
    }

@app.get("/debug")
async def debug_info():
    return {
        "status": "ok",
        "data_loaded": app_state.is_loaded,
        "memory_usage_mb": psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024),
        "system_state": {
            "id_to_name_size": len(app_state.recommender.id_to_name) if app_state.recommender else 0,
            "name_to_id_size": len(app_state.recommender.name_to_id) if app_state.recommender else 0,
            "sample_names": list(app_state.recommender.name_to_id.keys())[:5] if app_state.recommender else []
        },
        "file_info": {
            "base_dir": os.path.dirname(os.path.abspath(__file__)),
            "csv_exists": os.path.exists(os.path.join(os.path.dirname(os.path.abspath(__file__)), "productos.csv")),
            "directory_contents": os.listdir(os.path.dirname(os.path.abspath(__file__)))
        }
    }

if __name__ == "__main__":
    try:
        port = int(os.environ.get("PORT", 10000))
        logger.info(f"Starting server on port {port}")
        import uvicorn
        uvicorn.run(
            app, 
            host="0.0.0.0", 
            port=port,
            log_level="info"
        )
    except Exception as e:
        logger.error(f"Error starting server: {e}")
        logger.error(traceback.format_exc())
