import os
import gc
import psutil
import numpy as np
import pandas as pd
import tensorflow as tf
from typing import Dict, List
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from sklearn.preprocessing import LabelEncoder
from contextlib import asynccontextmanager
import logging
import tempfile
from pydantic import BaseModel

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Constants
MEMORY_LIMIT = 450  # MB
CHUNK_SIZE = 500

class MemoryMonitor:
    @staticmethod
    def get_memory_usage_mb() -> float:
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / (1024 * 1024)

    @staticmethod
    def log_memory(message: str):
        memory_mb = MemoryMonitor.get_memory_usage_mb()
        logger.info(f"{message} - Memory: {memory_mb:.2f} MB")

class EmbeddingStore:
    def __init__(self, batch_size: int = 100):
        self.batch_size = batch_size
        self.embeddings = {}
        self._current_batch = []
    
    def add_embedding(self, idx: int, embedding: np.ndarray):
        self.embeddings[idx] = embedding.astype(np.float16)  # Use float16 for memory efficiency
        
        if len(self.embeddings) % self.batch_size == 0:
            gc.collect()
    
    def get_similarity(self, query_id: int) -> Dict[int, float]:
        if query_id not in self.embeddings:
            raise ValueError(f"Query ID {query_id} not found")
            
        query_embedding = self.embeddings[query_id]
        similarities = {}
        
        # Process in batches to reduce memory usage
        batch = []
        for idx, embedding in self.embeddings.items():
            if idx != query_id:
                batch.append((idx, embedding))
                
                if len(batch) >= self.batch_size:
                    self._process_similarity_batch(query_embedding, batch, similarities)
                    batch = []
                    
        if batch:
            self._process_similarity_batch(query_embedding, batch, similarities)
            
        return similarities
    
    def _process_similarity_batch(self, query_embedding: np.ndarray, batch: List, similarities: Dict):
        batch_embeddings = np.stack([emb for _, emb in batch])
        batch_similarities = np.dot(batch_embeddings, query_embedding)
        
        for (idx, _), sim in zip(batch, batch_similarities):
            similarities[idx] = float(sim)

class RecommendationSystem:
    def __init__(self):
        self.id_to_name: Dict[int, str] = {}
        self.name_to_id: Dict[str, int] = {}
        self.embedding_store = EmbeddingStore()
        
    def get_recommendations(self, product_name: str, top_n: int = 5) -> List[Dict]:
        if product_name not in self.name_to_id:
            raise HTTPException(status_code=404, detail="Product not found")
            
        product_id = self.name_to_id[product_name]
        
        try:
            similarities = self.embedding_store.get_similarity(product_id)
            
            # Get top N recommendations
            sorted_items = sorted(
                similarities.items(),
                key=lambda x: x[1],
                reverse=True
            )[:top_n]
            
            return [
                {
                    "id": int(idx),
                    "name": self.id_to_name[idx],
                    "similarity": float(sim)
                }
                for idx, sim in sorted_items
            ]
            
        except Exception as e:
            logger.error(f"Error getting recommendations: {e}")
            raise HTTPException(status_code=500, detail="Error computing recommendations")

recommender = RecommendationSystem()

@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        MemoryMonitor.log_memory("Starting initialization")
        
        # Get paths
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        csv_path = os.path.join(BASE_DIR, "productos.csv")
        model_path = os.path.join(BASE_DIR, "recomendacion.tflite")
        
        # Process unique values
        unique_values = {"name": set(), "main_category": set(), "sub_category": set()}
        
        for chunk in pd.read_csv(csv_path, 
                               usecols=list(unique_values.keys()),
                               dtype=str,
                               chunksize=CHUNK_SIZE):
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
        recommender.id_to_name = dict(enumerate(encoders["name"].classes_))
        recommender.name_to_id = {name: idx for idx, name in recommender.id_to_name.items()}
        
        # Load TFLite model efficiently
        interpreter = tf.lite.Interpreter(
            model_path=model_path,
            num_threads=1
        )
        interpreter.allocate_tensors()
        
        # Process data in chunks
        for chunk in pd.read_csv(csv_path, chunksize=CHUNK_SIZE, dtype=str):
            valid_mask = chunk['name'].isin(recommender.name_to_id)
            if not valid_mask.any():
                continue
                
            chunk = chunk[valid_mask]
            for _, row in chunk.iterrows():
                if MemoryMonitor.get_memory_usage_mb() > MEMORY_LIMIT:
                    logger.warning("Memory limit approaching, triggering garbage collection")
                    gc.collect()
                    
                product_id = recommender.name_to_id[row['name']]
                
                # Get encoded values
                main_cat_id = encoders["main_category"].transform([row['main_category']])[0]
                sub_cat_id = encoders["sub_category"].transform([row['sub_category']])[0]
                
                # Get embeddings
                embeddings = []
                for detail in interpreter.get_tensor_details():
                    if "embedding" in detail["name"]:
                        embeddings.append(
                            interpreter.get_tensor(detail["index"])[
                                product_id if "product" in detail["name"]
                                else main_cat_id if "main" in detail["name"]
                                else sub_cat_id
                            ]
                        )
                
                # Combine and normalize embedding
                combined = np.concatenate(embeddings)
                normalized = combined / (np.linalg.norm(combined) + 1e-8)
                
                # Store embedding
                recommender.embedding_store.add_embedding(product_id, normalized)
                
            del chunk
            gc.collect()
        
        # Clean up
        del interpreter
        gc.collect()
        
        MemoryMonitor.log_memory("Initialization complete")
        
    except Exception as e:
        logger.error(f"Startup error: {e}")
        raise
    
    yield

# Create FastAPI app
app = FastAPI(
    title="Product Recommendation API",
    description="Memory-optimized product recommendation system",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    """Root endpoint for health checks"""
    return {
        "status": "ok",
        "memory_usage_mb": MemoryMonitor.get_memory_usage_mb()
    }

@app.get("/health")
async def health_check():
    """Detailed health check endpoint"""
    return {
        "status": "healthy",
        "memory_usage_mb": MemoryMonitor.get_memory_usage_mb(),
        "memory_limit_mb": MEMORY_LIMIT,
        "num_products": len(recommender.id_to_name)
    }

@app.get("/recomendaciones")
async def get_recommendations(product_name: str, top_n: int = 5):
    """Get product recommendations"""
    return {
        "producto": product_name,
        "recomendaciones": recommender.get_recommendations(product_name, top_n)
    }

if __name__ == "__main__":
    # Get port from environment variable with fallback to Render's default
    port = int(os.environ.get("PORT", 10000))
    
    # Log the port we're using
    logger.info(f"Starting server on port {port}")
    
    # Run the server
    import uvicorn
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=port,
        log_level="info",
        workers=1
    )
