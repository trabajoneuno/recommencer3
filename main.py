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
import mmap
import tempfile
import logging
import signal

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Memory limit in MB
MEMORY_LIMIT = 450

class MemoryGuard:
    @staticmethod
    def check_memory():
        process = psutil.Process(os.getpid())
        memory_mb = process.memory_info().rss / (1024 * 1024)
        if memory_mb > MEMORY_LIMIT:
            logger.error(f"Memory limit exceeded: {memory_mb:.2f} MB")
            # Force garbage collection
            gc.collect()
            return False
        return True

    @staticmethod
    def log_memory(message: str):
        process = psutil.Process(os.getpid())
        memory_mb = process.memory_info().rss / (1024 * 1024)
        logger.info(f"{message} - Memory: {memory_mb:.2f} MB")

class EmbeddingStore:
    def __init__(self, embedding_size: int):
        self.embedding_size = embedding_size
        self.current_batch = []
        self.batch_size = 100
        self.temp_files = []
        
    def add_embedding(self, idx: int, embedding: np.ndarray):
        self.current_batch.append((idx, embedding))
        if len(self.current_batch) >= self.batch_size:
            self._save_batch()
    
    def _save_batch(self):
        if not self.current_batch:
            return
            
        temp_file = tempfile.NamedTemporaryFile(delete=False)
        self.temp_files.append(temp_file.name)
        
        # Save as compressed npz
        data = dict(enumerate(self.current_batch))
        np.savez_compressed(temp_file, **data)
        
        self.current_batch = []
        gc.collect()

    def get_similarity(self, query_id: int, target_ids: List[int]) -> np.ndarray:
        query_embedding = None
        target_embeddings = []
        
        # Load embeddings from temp files
        for temp_file in self.temp_files:
            data = np.load(temp_file)
            for key in data.files:
                idx, embedding = data[key]
                if idx == query_id:
                    query_embedding = embedding
                elif idx in target_ids:
                    target_embeddings.append((idx, embedding))
                    
        if query_embedding is None:
            raise ValueError(f"Query ID {query_id} not found")
            
        # Calculate similarities
        similarities = {}
        for idx, emb in target_embeddings:
            sim = np.dot(query_embedding, emb)
            similarities[idx] = sim
            
        return similarities

    def cleanup(self):
        for temp_file in self.temp_files:
            try:
                os.unlink(temp_file)
            except Exception as e:
                logger.error(f"Error cleaning up temp file {temp_file}: {e}")
        self.temp_files = []

class RecommendationSystem:
    def __init__(self):
        self.id_to_name: Dict[int, str] = {}
        self.name_to_id: Dict[str, int] = {}
        self.embedding_store = None
        
    def get_recommendations(self, product_name: str, top_n: int = 5) -> List[Dict]:
        if not MemoryGuard.check_memory():
            raise HTTPException(status_code=503, detail="System under high memory load")
            
        if product_name not in self.name_to_id:
            raise HTTPException(status_code=404, detail="Product not found")
            
        product_id = self.name_to_id[product_name]
        all_ids = list(self.id_to_name.keys())
        
        try:
            similarities = self.embedding_store.get_similarity(product_id, all_ids)
            
            # Get top N recommendations
            sorted_ids = sorted(similarities.keys(), 
                              key=lambda x: similarities[x], 
                              reverse=True)[:top_n]
            
            return [
                {
                    "id": int(idx),
                    "name": self.id_to_name[idx],
                    "similarity": float(similarities[idx])
                }
                for idx in sorted_ids
                if idx != product_id
            ]
            
        except Exception as e:
            logger.error(f"Error getting recommendations: {e}")
            raise HTTPException(status_code=500, detail="Error computing recommendations")

recommender = RecommendationSystem()

@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        MemoryGuard.log_memory("Starting initialization")
        
        # Get paths
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        csv_path = os.path.join(BASE_DIR, "productos.csv")
        model_path = os.path.join(BASE_DIR, "recomendacion.tflite")
        
        # Process unique values with minimal memory
        unique_values = {"name": set(), "main_category": set(), "sub_category": set()}
        chunk_size = 500
        
        for chunk in pd.read_csv(csv_path, 
                               usecols=list(unique_values.keys()),
                               dtype=str,
                               chunksize=chunk_size):
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
            num_threads=1  # Reduce memory usage
        )
        interpreter.allocate_tensors()
        
        # Get embedding dimensions from model
        embedding_size = sum(
            interpreter.get_tensor(
                detail["index"]
            ).shape[1]
            for detail in interpreter.get_tensor_details()
            if "embedding" in detail["name"]
        )
        
        # Initialize embedding store
        recommender.embedding_store = EmbeddingStore(embedding_size)
        
        # Process data in small chunks
        for chunk in pd.read_csv(csv_path, chunksize=chunk_size, dtype=str):
            valid_mask = chunk['name'].isin(recommender.name_to_id)
            if not valid_mask.any():
                continue
                
            chunk = chunk[valid_mask]
            for _, row in chunk.iterrows():
                if not MemoryGuard.check_memory():
                    raise Exception("Memory limit exceeded during processing")
                    
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
        
        MemoryGuard.log_memory("Initialization complete")
        
    except Exception as e:
        logger.error(f"Startup error: {e}")
        raise
    
    yield
    
    # Cleanup
    if recommender.embedding_store:
        recommender.embedding_store.cleanup()
    gc.collect()
    MemoryGuard.log_memory("Shutdown complete")

app = FastAPI(lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
async def health_check():
    memory_mb = psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)
    return {
        "status": "healthy",
        "memory_usage_mb": memory_mb,
        "memory_limit_mb": MEMORY_LIMIT
    }

@app.get("/recomendaciones")
async def get_recommendations(product_name: str, top_n: int = 5):
    return {
        "producto": product_name,
        "recomendaciones": recommender.get_recommendations(product_name, top_n)
    }

def handle_exit(signum, frame):
    logger.info("Received shutdown signal, cleaning up...")
    if recommender.embedding_store:
        recommender.embedding_store.cleanup()
    sys.exit(0)

if __name__ == "__main__":
    # Register signal handlers
    signal.signal(signal.SIGINT, handle_exit)
    signal.signal(signal.SIGTERM, handle_exit)
    
    # Get port from environment with fallback
    port = int(os.environ.get("PORT", 8000))
    
    # Configure uvicorn with minimal settings
    import uvicorn
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=port,
        workers=1,  # Minimize memory usage
        loop="auto",
        limit_concurrency=10,  # Limit concurrent requests
        limit_max_requests=1000  # Restart worker after N requests
    )
