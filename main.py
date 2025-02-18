import os
import gc
import psutil
import numpy as np
import pandas as pd
import tensorflow as tf
from typing import Dict, List, Optional, Generator
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from sklearn.preprocessing import LabelEncoder
from contextlib import asynccontextmanager
import mmap
import tempfile
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MemoryMonitor:
    @staticmethod
    def get_memory_usage_mb() -> float:
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / (1024 * 1024)

    @staticmethod
    def log_memory(message: str):
        logger.info(f"{message} - Memory usage: {MemoryMonitor.get_memory_usage_mb():.2f} MB")

class MemoryMappedEmbeddings:
    def __init__(self, shape: tuple, dtype=np.float32):
        self.shape = shape
        self.dtype = dtype
        self.temp_file = None
        self.mmap_array = None

    def initialize(self):
        # Create temporary file for memory mapping
        self.temp_file = tempfile.NamedTemporaryFile(delete=False)
        # Create empty file of required size
        self.temp_file.write(b'\0' * (np.prod(self.shape) * np.dtype(self.dtype).itemsize))
        self.temp_file.flush()
        
        # Create memory-mapped array
        self.mmap_array = np.memmap(
            self.temp_file.name,
            dtype=self.dtype,
            mode='r+',
            shape=self.shape
        )

    def __del__(self):
        if self.mmap_array is not None:
            self.mmap_array._mmap.close()
        if self.temp_file:
            os.unlink(self.temp_file.name)

class StreamingCSVProcessor:
    def __init__(self, chunk_size: int = 1000):
        self.chunk_size = chunk_size

    def process_unique_values(self, csv_path: str) -> tuple:
        names, main_cats, sub_cats = set(), set(), set()
        
        for chunk in pd.read_csv(
            csv_path, 
            usecols=["name", "main_category", "sub_category"],
            dtype=str,
            chunksize=self.chunk_size
        ):
            names.update(chunk['name'].dropna().unique())
            main_cats.update(chunk['main_category'].dropna().unique())
            sub_cats.update(chunk['sub_category'].dropna().unique())
            
            del chunk
            gc.collect()
        
        return names, main_cats, sub_cats

class RecommendationSystem:
    def __init__(self):
        self.id_to_name: Dict[int, str] = {}
        self.name_to_id: Dict[str, int] = {}
        self.embeddings: Optional[MemoryMappedEmbeddings] = None
        self._mini_batch_size = 100  # Size for processing embeddings in batches
        
    def _process_embeddings_batch(self, query_id: int, start_idx: int, batch_size: int) -> np.ndarray:
        """Process similarity computation in small batches"""
        end_idx = min(start_idx + batch_size, self.embeddings.shape[0])
        query_embedding = self.embeddings.mmap_array[query_id]
        batch_similarities = np.dot(self.embeddings.mmap_array[start_idx:end_idx], query_embedding)
        return batch_similarities

    def get_recommendations(self, product_name: str, top_n: int = 5) -> List[Dict]:
        if product_name not in self.name_to_id:
            raise HTTPException(status_code=404, detail="Product not found")
        
        product_id = self.name_to_id[product_name]
        num_products = self.embeddings.shape[0]
        
        # Process similarities in batches to reduce memory usage
        all_similarities = []
        for start_idx in range(0, num_products, self._mini_batch_size):
            batch_similarities = self._process_embeddings_batch(
                product_id, start_idx, self._mini_batch_size
            )
            all_similarities.extend(batch_similarities)
        
        # Convert to numpy array only for top-k selection
        similarities = np.array(all_similarities)
        similarities[product_id] = -np.inf
        
        # Get top-k indices
        top_indices = np.argpartition(-similarities, top_n)[:top_n]
        top_indices = top_indices[np.argsort(-similarities[top_indices])]
        
        return [
            {
                "id": int(idx),
                "name": self.id_to_name[int(idx)],
                "similarity": float(similarities[idx])
            }
            for idx in top_indices
        ]

recommender = RecommendationSystem()

@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        MemoryMonitor.log_memory("Starting initialization")
        
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        csv_path = os.path.join(BASE_DIR, "productos.csv")
        model_path = os.path.join(BASE_DIR, "recomendacion.tflite")
        
        # Process CSV in streaming fashion
        csv_processor = StreamingCSVProcessor()
        names, main_cats, sub_cats = csv_processor.process_unique_values(csv_path)
        
        # Create encoders
        name_encoder = LabelEncoder().fit(list(names))
        main_cat_encoder = LabelEncoder().fit(list(main_cats))
        sub_cat_encoder = LabelEncoder().fit(list(sub_cats))
        
        # Set up ID mappings
        recommender.id_to_name = dict(enumerate(name_encoder.classes_))
        recommender.name_to_id = {name: idx for idx, name in recommender.id_to_name.items()}
        
        # Load TFLite model
        interpreter = tf.lite.Interpreter(model_path=model_path)
        interpreter.allocate_tensors()
        
        def get_embedding_matrix(layer_name: str) -> np.ndarray:
            for detail in interpreter.get_tensor_details():
                if layer_name in detail["name"]:
                    return interpreter.get_tensor(detail["index"])
            raise ValueError(f"Layer {layer_name} not found")
        
        # Get embedding dimensions
        prod_emb = get_embedding_matrix("product_embedding")
        main_emb = get_embedding_matrix("main_cat_embedding")
        sub_emb = get_embedding_matrix("sub_cat_embedding")
        
        combined_dim = prod_emb.shape[1] + main_emb.shape[1] + sub_emb.shape[1]
        
        # Initialize memory-mapped embeddings
        recommender.embeddings = MemoryMappedEmbeddings(
            shape=(len(names), combined_dim),
            dtype=np.float32
        )
        recommender.embeddings.initialize()
        
        # Process data in chunks to populate embeddings
        chunk_size = 1000
        for chunk in pd.read_csv(csv_path, chunksize=chunk_size, dtype=str):
            valid_mask = chunk['name'].isin(recommender.name_to_id)
            if not valid_mask.any():
                continue
                
            chunk = chunk[valid_mask]
            product_ids = chunk['name'].map(recommender.name_to_id).values
            
            # Get embeddings for this chunk
            main_cat_ids = main_cat_encoder.transform(chunk['main_category'])
            sub_cat_ids = sub_cat_encoder.transform(chunk['sub_category'])
            
            # Combine embeddings
            chunk_embeddings = np.concatenate([
                prod_emb[product_ids],
                main_emb[main_cat_ids],
                sub_emb[sub_cat_ids]
            ], axis=1)
            
            # Normalize chunk embeddings
            norms = np.linalg.norm(chunk_embeddings, axis=1, keepdims=True)
            chunk_embeddings /= (norms + 1e-8)
            
            # Update memory-mapped array
            recommender.embeddings.mmap_array[product_ids] = chunk_embeddings
            
            del chunk_embeddings
            gc.collect()
        
        # Clean up
        del interpreter, prod_emb, main_emb, sub_emb
        gc.collect()
        
        MemoryMonitor.log_memory("Initialization complete")
        
    except Exception as e:
        logger.error(f"Startup error: {str(e)}")
        raise
    
    yield
    
    # Cleanup
    if recommender.embeddings:
        del recommender.embeddings
    gc.collect()
    MemoryMonitor.log_memory("Shutdown complete")

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
    return {
        "status": "healthy",
        "memory_usage_mb": MemoryMonitor.get_memory_usage_mb()
    }

@app.get("/recomendaciones")
async def get_recommendations(product_name: str, top_n: int = 5):
    return {
        "producto": product_name,
        "recomendaciones": recommender.get_recommendations(product_name, top_n)
    }

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=port)

