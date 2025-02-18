import os
import gc
import psutil
import numpy as np
import pandas as pd
import tensorflow as tf
from typing import Dict, List, Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from sklearn.preprocessing import LabelEncoder
from contextlib import asynccontextmanager
import logging
import tempfile
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Simple FastAPI app - this worked with port binding
app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class MemoryOptimizedStorage:
    def __init__(self, chunk_size: int = 100):
        self.chunk_size = chunk_size
        self.temp_dir = tempfile.mkdtemp()
        self.current_chunk = {}
        self.chunk_files = []
        logger.info(f"Created temporary directory: {self.temp_dir}")
    
    def add_data(self, idx: int, embedding: np.ndarray):
        """Add data to current chunk, save if chunk is full"""
        self.current_chunk[idx] = embedding.astype(np.float16)
        
        if len(self.current_chunk) >= self.chunk_size:
            self._save_chunk()
            
    def _save_chunk(self):
        """Save current chunk to disk"""
        if not self.current_chunk:
            return
            
        chunk_path = Path(self.temp_dir) / f"chunk_{len(self.chunk_files)}.npz"
        np.savez_compressed(
            chunk_path,
            indices=np.array(list(self.current_chunk.keys())),
            embeddings=np.stack(list(self.current_chunk.values()))
        )
        self.chunk_files.append(str(chunk_path))
        self.current_chunk.clear()
        gc.collect()
    
    def get_similarities(self, query_id: int) -> Dict[int, float]:
        """Calculate similarities efficiently"""
        similarities = {}
        query_embedding = None
        
        # Find query embedding
        for chunk_file in self.chunk_files:
            chunk = np.load(chunk_file)
            indices = chunk['indices']
            if query_id in indices:
                idx = np.where(indices == query_id)[0][0]
                query_embedding = chunk['embeddings'][idx]
                break
                
        if query_embedding is None and query_id in self.current_chunk:
            query_embedding = self.current_chunk[query_id]
            
        if query_embedding is None:
            raise ValueError(f"Query ID {query_id} not found")
            
        # Calculate similarities chunk by chunk
        for chunk_file in self.chunk_files:
            chunk = np.load(chunk_file)
            chunk_sims = np.dot(chunk['embeddings'], query_embedding)
            for idx, sim in zip(chunk['indices'], chunk_sims):
                if idx != query_id:
                    similarities[int(idx)] = float(sim)
                    
        # Process current chunk
        for idx, emb in self.current_chunk.items():
            if idx != query_id:
                similarities[idx] = float(np.dot(emb, query_embedding))
                
        return similarities
    
    def cleanup(self):
        """Remove temporary files"""
        for file_path in self.chunk_files:
            try:
                os.remove(file_path)
            except Exception as e:
                logger.error(f"Error removing {file_path}: {e}")
        try:
            os.rmdir(self.temp_dir)
        except Exception as e:
            logger.error(f"Error removing temp dir: {e}")

class RecommendationSystem:
    def __init__(self):
        self.id_to_name: Dict[int, str] = {}
        self.name_to_id: Dict[str, int] = {}
        self.storage: Optional[MemoryOptimizedStorage] = None
        
    def get_recommendations(self, product_name: str, top_n: int = 5) -> List[Dict]:
        """Get recommendations with memory optimization"""
        if product_name not in self.name_to_id:
            raise HTTPException(status_code=404, detail="Product not found")
            
        product_id = self.name_to_id[product_name]
        try:
            similarities = self.storage.get_similarities(product_id)
            
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

recommender = RecommendationSystem()

@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        # Log startup with port information
        port = os.environ.get("PORT", "10000")
        logger.info(f"Starting initialization on port {port}")
        
        # Get file paths
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        csv_path = os.path.join(BASE_DIR, "productos.csv")
        model_path = os.path.join(BASE_DIR, "recomendacion.tflite")
        
        # Initialize storage
        recommender.storage = MemoryOptimizedStorage(chunk_size=100)
        
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
            num_threads=1  # Minimize memory usage
        )
        interpreter.allocate_tensors()
        
        # Process data in chunks
        for chunk in pd.read_csv(csv_path, chunksize=chunk_size, dtype=str):
            valid_mask = chunk['name'].isin(recommender.name_to_id)
            if not valid_mask.any():
                continue
                
            chunk = chunk[valid_mask]
            for _, row in chunk.iterrows():
                product_id = recommender.name_to_id[row['name']]
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
                recommender.storage.add_data(product_id, normalized)
            
            del chunk
            gc.collect()
        
        # Save final chunk and clean up
        recommender.storage._save_chunk()
        del interpreter
        gc.collect()
        
        logger.info("Initialization complete")
        logger.info(f"Server ready to accept requests on port {port}")
        
    except Exception as e:
        logger.error(f"Startup error: {e}")
        raise
    
    yield
    
    # Cleanup on shutdown
    if recommender.storage:
        recommender.storage.cleanup()

# Configure FastAPI with lifespan
app = FastAPI(lifespan=lifespan)

@app.get("/")
async def root():
    """Root endpoint for health checks"""
    memory_mb = psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)
    return {
        "status": "ok",
        "memory_usage_mb": memory_mb,
        "port": os.environ.get("PORT", "10000")
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    memory_mb = psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)
    return {
        "status": "healthy",
        "memory_usage_mb": memory_mb,
        "port": os.environ.get("PORT", "10000"),
        "num_products": len(recommender.id_to_name)
    }

@app.get("/recomendaciones")
async def get_recommendations(product_name: str, top_n: int = 5):
    """Get product recommendations"""
    return {
        "producto": product_name,
        "recomendaciones": recommender.get_recommendations(product_name, top_n)
    }
