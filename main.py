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
import mmap
import tempfile

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Create the FastAPI app first - this is what worked for port binding
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
async def startup_event():
    port = os.environ.get("PORT", 10000)
    logger.info(f"Application startup on port {port}")

class MemoryMappedStore:
    def __init__(self):
        self.temp_file = None
        self.mmap_array = None
        self.index_map = {}
        self.current_index = 0
        
    def initialize(self, embedding_size: int, max_items: int):
        # Create temporary file
        self.temp_file = tempfile.NamedTemporaryFile(delete=False)
        array_size = max_items * embedding_size * 4  # 4 bytes per float32
        self.temp_file.write(b'\0' * array_size)
        self.temp_file.flush()
        
        # Create memory-mapped array
        self.mmap_array = np.memmap(
            self.temp_file.name,
            dtype=np.float32,
            mode='r+',
            shape=(max_items, embedding_size)
        )
        
    def add_embedding(self, idx: int, embedding: np.ndarray):
        if idx not in self.index_map:
            self.index_map[idx] = self.current_index
            self.current_index += 1
        
        mapped_idx = self.index_map[idx]
        self.mmap_array[mapped_idx] = embedding
        self.mmap_array.flush()
        
    def get_similarities(self, query_idx: int, batch_size: int = 100) -> Dict[int, float]:
        if query_idx not in self.index_map:
            raise ValueError(f"Query ID {query_idx} not found")
            
        mapped_query_idx = self.index_map[query_idx]
        query_embedding = self.mmap_array[mapped_query_idx]
        
        similarities = {}
        for start_idx in range(0, self.current_index, batch_size):
            end_idx = min(start_idx + batch_size, self.current_index)
            batch_similarities = np.dot(self.mmap_array[start_idx:end_idx], query_embedding)
            
            for i, sim in enumerate(batch_similarities):
                real_idx = next(k for k, v in self.index_map.items() if v == start_idx + i)
                if real_idx != query_idx:
                    similarities[real_idx] = float(sim)
        
        return similarities
    
    def cleanup(self):
        if self.mmap_array is not None:
            self.mmap_array._mmap.close()
        if self.temp_file:
            os.unlink(self.temp_file.name)

class RecommendationSystem:
    def __init__(self):
        self.id_to_name: Dict[int, str] = {}
        self.name_to_id: Dict[str, int] = {}
        self.embedding_store = MemoryMappedStore()
        
    def get_recommendations(self, product_name: str, top_n: int = 5) -> List[Dict]:
        if product_name not in self.name_to_id:
            raise HTTPException(status_code=404, detail="Product not found")
        
        product_id = self.name_to_id[product_name]
        try:
            similarities = self.embedding_store.get_similarities(product_id)
            top_ids = sorted(similarities.keys(), key=lambda x: similarities[x], reverse=True)[:top_n]
            
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
        logger.info("Starting initialization")
        
        # Get file paths
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        csv_path = os.path.join(BASE_DIR, "productos.csv")
        model_path = os.path.join(BASE_DIR, "recomendacion.tflite")
        
        # Count total products for initialization
        total_products = 0
        chunk_size = 500
        for chunk in pd.read_csv(csv_path, usecols=['name'], chunksize=chunk_size):
            total_products += len(chunk['name'].unique())
            del chunk
            gc.collect()
        
        # Process unique values
        unique_values = {"name": set(), "main_category": set(), "sub_category": set()}
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
        interpreter = tf.lite.Interpreter(model_path=model_path, num_threads=1)
        interpreter.allocate_tensors()
        
        # Get embedding size
        embedding_tensors = [
            interpreter.get_tensor(detail["index"])
            for detail in interpreter.get_tensor_details()
            if "embedding" in detail["name"]
        ]
        total_embedding_size = sum(tensor.shape[1] for tensor in embedding_tensors)
        
        # Initialize embedding store
        recommender.embedding_store.initialize(
            embedding_size=total_embedding_size,
            max_items=total_products
        )
        
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
                recommender.embedding_store.add_embedding(product_id, normalized)
            
            del chunk
            gc.collect()
        
        # Clean up TF interpreter
        del interpreter
        gc.collect()
        
        logger.info("Initialization complete")
        
    except Exception as e:
        logger.error(f"Startup error: {e}")
        raise
    
    yield
    
    # Cleanup on shutdown
    recommender.embedding_store.cleanup()

# Add lifespan to app
app = FastAPI(lifespan=lifespan)

@app.get("/")
async def root():
    """Root endpoint for health checks"""
    memory_mb = psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)
    return {
        "status": "ok",
        "memory_usage_mb": memory_mb
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    memory_mb = psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)
    return {
        "status": "healthy",
        "memory_usage_mb": memory_mb,
        "num_products": len(recommender.id_to_name)
    }

@app.get("/recomendaciones")
async def get_recommendations(product_name: str, top_n: int = 5):
    """Get product recommendations"""
    return {
        "producto": product_name,
        "recomendaciones": recommender.get_recommendations(product_name, top_n)
    }
