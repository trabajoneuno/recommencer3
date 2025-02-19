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

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class MemoryEfficientStore:
    def __init__(self, chunk_size: int = 100):
        self.chunk_size = chunk_size
        self.temp_dir = Path(tempfile.mkdtemp())
        self.chunks = []
        self.current_chunk = {}
        
    def add_embedding(self, idx: int, embedding: np.ndarray):
        self.current_chunk[idx] = embedding.astype(np.float16)
        
        if len(self.current_chunk) >= self.chunk_size:
            self._save_chunk()
    
    def _save_chunk(self):
        if not self.current_chunk:
            return
            
        chunk_file = self.temp_dir / f"chunk_{len(self.chunks)}.npz"
        np.savez_compressed(
            chunk_file,
            indices=np.array(list(self.current_chunk.keys())),
            embeddings=np.stack(list(self.current_chunk.values()))
        )
        self.chunks.append(chunk_file)
        self.current_chunk.clear()
        gc.collect()
    
    def get_similarities(self, query_id: int) -> Dict[int, float]:
        query_embedding = None
        
        # Find query embedding
        for chunk_file in self.chunks:
            data = np.load(chunk_file)
            idx = np.where(data['indices'] == query_id)[0]
            if len(idx) > 0:
                query_embedding = data['embeddings'][idx[0]]
                break
        
        if query_embedding is None and query_id in self.current_chunk:
            query_embedding = self.current_chunk[query_id]
        
        if query_embedding is None:
            raise ValueError(f"Query ID {query_id} not found")
        
        # Calculate similarities
        similarities = {}
        for chunk_file in self.chunks:
            data = np.load(chunk_file)
            chunk_sims = np.dot(data['embeddings'], query_embedding)
            for idx, sim in zip(data['indices'], chunk_sims):
                if idx != query_id:
                    similarities[int(idx)] = float(sim)
        
        # Process current chunk
        for idx, emb in self.current_chunk.items():
            if idx != query_id:
                similarities[idx] = float(np.dot(emb, query_embedding))
        
        return similarities
    
    def cleanup(self):
        for chunk_file in self.chunks:
            try:
                chunk_file.unlink()
            except Exception as e:
                logger.error(f"Error removing {chunk_file}: {e}")
        try:
            self.temp_dir.rmdir()
        except Exception as e:
            logger.error(f"Error removing temp dir: {e}")

class RecommendationSystem:
    def __init__(self):
        self.id_to_name: Dict[int, str] = {}
        self.name_to_id: Dict[str, int] = {}
        self.embedding_store = MemoryEfficientStore()
    
    def process_data(self, csv_path: str):
        chunk_size = 500
        
        # Modificar para usar main_category en lugar de category
        unique_values = {"name": set(), "main_category": set()}
        for chunk in pd.read_csv(csv_path, usecols=list(unique_values.keys()), 
                               dtype=str, chunksize=chunk_size):
            for col in unique_values:
                unique_values[col].update(chunk[col].dropna().unique())
            del chunk
            gc.collect()
        
        # Create encoders
        encoders = {
            col: LabelEncoder().fit(list(values))
            for col, values in unique_values.items()
        }
        
        # Set up ID mappings
        self.id_to_name = dict(enumerate(encoders["name"].classes_))
        self.name_to_id = {name: idx for idx, name in self.id_to_name.items()}
        
        # Process data in chunks
        for chunk in pd.read_csv(csv_path, chunksize=chunk_size, dtype=str):
            valid_mask = chunk['name'].isin(self.name_to_id)
            if not valid_mask.any():
                continue
            
            chunk = chunk[valid_mask]
            for _, row in chunk.iterrows():
                product_id = self.name_to_id[row['name']]
                
                # Modificar para usar main_category
                category_id = encoders["main_category"].transform([row['main_category']])[0]
                embedding = np.zeros(50, dtype=np.float32)
                embedding[category_id % 50] = 1
                
                # Store embedding
                self.embedding_store.add_embedding(product_id, embedding)
    
    def get_recommendations(self, product_name: str, top_n: int = 5) -> List[Dict]:
        if product_name not in self.name_to_id:
            raise HTTPException(status_code=404, detail="Product not found")
        
        product_id = self.name_to_id[product_name]
        try:
            similarities = self.embedding_store.get_similarities(product_id)
            
            # Get top N recommendations
            top_ids = sorted(
                similarities.keys(),
                key=lambda x: similarities[x],
                reverse=True
            )[:top_n]
            
            return [
                {
                    "id": int(idx),
                    "name": self.id_to_name[idx],
                    "similarity": float(similarities[idx])
                }
                for idx in top_ids
            ]
            
        except Exception as e:
            logger.error(f"Error getting recommendations: {e}")
            raise HTTPException(status_code=500, detail="Error computing recommendations")

# Initialize recommender
recommender = RecommendationSystem()

@app.on_event("startup")
async def startup_event():
    try:
        # Get file path
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        csv_path = os.path.join(BASE_DIR, "productos.csv")
        
        logger.info("Starting data processing")
        recommender.process_data(csv_path)
        logger.info("Data processing complete")
        
    except Exception as e:
        logger.error(f"Error during startup: {e}")
        logger.error(traceback.format_exc())

@app.on_event("shutdown")
async def shutdown_event():
    logger.info("Shutting down")
    recommender.embedding_store.cleanup()

@app.get("/")
async def root():
    memory_mb = psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)
    return {
        "status": "ok",
        "memory_usage_mb": memory_mb
    }

@app.get("/health")
async def health_check():
    memory_mb = psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)
    return {
        "status": "healthy",
        "memory_usage_mb": memory_mb,
        "num_products": len(recommender.id_to_name)
    }

@app.get("/recomendaciones")
async def get_recommendations(product_name: str, top_n: int = 5):
    return {
        "producto": product_name,
        "recomendaciones": recommender.get_recommendations(product_name, top_n)
    }

@app.get("/productos")
async def list_products(
    page: Optional[int] = 1,
    page_size: Optional[int] = 50,
    search: Optional[str] = None
):
    """
    List available products with optional pagination and search.
    """
    if page < 1:
        raise HTTPException(status_code=400, detail="Page number must be >= 1")
    if page_size < 1 or page_size > 100:
        raise HTTPException(status_code=400, detail="Page size must be between 1 and 100")
    
    products = list(recommender.name_to_id.keys())
    
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
    """Endpoint para diagnóstico del sistema"""
    try:
        return {
            "status": "ok",
            "memory_usage_mb": psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024),
            "system_state": {
                "id_to_name_size": len(recommender.id_to_name),
                "name_to_id_size": len(recommender.name_to_id),
                "sample_names": list(recommender.name_to_id.keys())[:5]
            },
            "file_info": {
                "base_dir": os.path.dirname(os.path.abspath(__file__)),
                "csv_exists": os.path.exists(os.path.join(os.path.dirname(os.path.abspath(__file__)), "productos.csv")),
                "directory_contents": os.listdir(os.path.dirname(os.path.abspath(__file__)))
            }
        }
    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
            "traceback": traceback.format_exc()
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
