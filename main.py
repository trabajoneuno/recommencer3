import os
import gc
import psutil
import numpy as np
import pandas as pd
import tensorflow as tf
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from sklearn.preprocessing import LabelEncoder
from contextlib import asynccontextmanager

# For caching recommendations
from functools import lru_cache

def get_memory_usage_mb():
    process = psutil.Process(os.getpid())
    mem_bytes = process.memory_info().rss
    return mem_bytes / (1024 * 1024)

print(f"Initial memory usage: {get_memory_usage_mb():.2f} MB")

# Global variables
id_to_name = None
name_to_id = None
num_products = None
product_meta = None
combined_embeddings_norm = None

# A simple dictionary cache for recommendations
recommendation_cache = {}

# Define base directory (assumes CSV and model are in the same folder as this file)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

@asynccontextmanager
async def lifespan(app: FastAPI):
    global id_to_name, name_to_id, num_products, product_meta, combined_embeddings_norm

    print("Starting startup...")
    try:
        # Paths to files
        csv_path = os.path.join(BASE_DIR, "productos.csv")
        model_path = os.path.join(BASE_DIR, "recomendacion.tflite")
        
        # === First Pass: Gather Unique Values ===
        usecols = ["name", "main_category", "sub_category"]
        all_names = set()
        all_main_categories = set()
        all_sub_categories = set()
        
        chunk_size = 1000
        for chunk in pd.read_csv(csv_path, usecols=usecols, chunksize=chunk_size, dtype=str):
            all_names.update(chunk['name'].unique())
            all_main_categories.update(chunk['main_category'].unique())
            all_sub_categories.update(chunk['sub_category'].unique())
        
        print(f"Found {len(all_names)} unique products")
        print(f"Found {len(all_main_categories)} unique main categories")
        print(f"Found {len(all_sub_categories)} unique sub categories")
        
        # === Fit LabelEncoders ===
        name_enc = LabelEncoder()
        main_cat_enc = LabelEncoder()
        sub_cat_enc = LabelEncoder()
        name_enc.fit(list(all_names))
        main_cat_enc.fit(list(all_main_categories))
        sub_cat_enc.fit(list(all_sub_categories))
        
        # Create lookup dictionaries
        id_to_name = dict(zip(range(len(name_enc.classes_)), name_enc.classes_))
        name_to_id = {name: idx for idx, name in id_to_name.items()}
        num_products = len(name_enc.classes_)
        
        # Preallocate metadata array: column 0 for main_category and column 1 for sub_category
        product_meta = np.zeros((num_products, 2), dtype=np.int32)
        
        # === Second Pass: Process CSV in a Vectorized Manner ===
        for chunk in pd.read_csv(csv_path, usecols=usecols, chunksize=chunk_size, dtype=str):
            # Map names to their IDs
            ids = chunk['name'].map(name_to_id)
            # Skip rows without valid mapping (for safety)
            valid = ids.notnull()
            if valid.any():
                chunk = chunk[valid]
                ids = ids[valid].astype(np.int32).values
                main_encoded = main_cat_enc.transform(chunk['main_category'].values)
                sub_encoded = sub_cat_enc.transform(chunk['sub_category'].values)
                product_meta[ids, 0] = main_encoded
                product_meta[ids, 1] = sub_encoded
        
        print(f"Memory usage after metadata: {get_memory_usage_mb():.2f} MB")
        
        # === Load TFLite Model and Extract Embeddings ===
        interpreter = tf.lite.Interpreter(model_path=model_path)
        interpreter.allocate_tensors()
        print("TFLite model loaded.")

        def get_embedding_weights(layer_name):
            for tensor_detail in interpreter.get_tensor_details():
                if layer_name in tensor_detail["name"]:
                    tensor = interpreter.get_tensor(tensor_detail["index"])
                    if tensor.ndim >= 2:
                        return tensor.astype(np.float32)
            raise ValueError(f"Tensor for {layer_name} not found")
        
        # Get embeddings and convert to float16 for memory savings
        prod_weights = get_embedding_weights("product_embedding").astype(np.float16)
        main_weights = get_embedding_weights("main_cat_embedding").astype(np.float16)
        sub_weights = get_embedding_weights("sub_cat_embedding").astype(np.float16)
        
        # Release the interpreter and collect garbage
        del interpreter
        gc.collect()
        
        # === Precompute Combined Embeddings ===
        # We assume:
        # - prod_weights: shape (num_products, d_prod)
        # - main_weights: shape (num_main_categories, d_main)
        # - sub_weights: shape (num_sub_categories, d_sub)
        combined_embeddings = np.concatenate([
            prod_weights,
            main_weights[product_meta[:, 0]],
            sub_weights[product_meta[:, 1]]
        ], axis=1).astype(np.float16)
        
        # Normalize embeddings (temporarily cast to float32 for stability)
        norms = np.linalg.norm(combined_embeddings.astype(np.float32), axis=1, keepdims=True)
        combined_embeddings_norm = (combined_embeddings.astype(np.float32)) / (norms + 1e-10)
        
        # Free arrays that are no longer needed
        del prod_weights, main_weights, sub_weights, combined_embeddings
        gc.collect()
        
        print(f"Final memory usage after setup: {get_memory_usage_mb():.2f} MB")
        print("Startup completed. API is ready.")
    
    except Exception as e:
        print(f"Error during startup: {str(e)}")
        raise e
    
    yield
    print("Shutting down API...")

# Create FastAPI app and set up CORS
app = FastAPI(lifespan=lifespan, title="Product Recommendation API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def health_check():
    return {"status": "ok"}

def compute_recommendations(product_id: int, top_n: int):
    """
    Compute the top_n recommendations for a given product_id using cosine similarity.
    """
    query_embedding = combined_embeddings_norm[product_id]
    similarities = np.dot(combined_embeddings_norm, query_embedding)
    similarities[product_id] = -np.inf  # Exclude the product itself
    top_indices = np.argpartition(-similarities, top_n)[:top_n]
    top_indices = top_indices[np.argsort(-similarities[top_indices])]
    return top_indices.tolist()

@app.get("/recomendaciones")
def get_recomendaciones(product_name: str, top_n: int = 5):
    global id_to_name, name_to_id, recommendation_cache
    
    if product_name not in name_to_id:
        raise HTTPException(status_code=404, detail="Producto no encontrado")
    
    product_id = name_to_id[product_name]
    # Create a cache key based on product_id and top_n
    cache_key = (product_id, top_n)
    
    if cache_key in recommendation_cache:
        top_indices = recommendation_cache[cache_key]
    else:
        top_indices = compute_recommendations(product_id, top_n)
        recommendation_cache[cache_key] = top_indices
    
    recomendaciones = [
        {"id": int(pid), "name": id_to_name.get(int(pid), f"ID {pid}")}
        for pid in top_indices
    ]
    
    return {"producto": product_name, "recomendaciones": recomendaciones}

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=port)

