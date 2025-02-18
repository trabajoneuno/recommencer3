import os
import gc
import psutil
import numpy as np
import pandas as pd
import tensorflow as tf
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import LabelEncoder
from contextlib import asynccontextmanager

# Function to get memory usage
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
prod_weights = None
main_weights = None
sub_weights = None

# Define base directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

@asynccontextmanager
async def lifespan(app: FastAPI):
    global id_to_name, name_to_id, num_products, product_meta
    global prod_weights, main_weights, sub_weights

    print("Starting startup...")
    try:
        # File paths
        csv_path = os.path.join(BASE_DIR, "productos.csv")
        model_path = os.path.join(BASE_DIR, "recomendacion.tflite")

        # 1. First pass: Collect all unique values
        usecols = ["name", "main_category", "sub_category"]
        all_names = set()
        all_main_categories = set()
        all_sub_categories = set()
        
        chunk_size = 1000
        for chunk in pd.read_csv(csv_path, usecols=usecols, chunksize=chunk_size):
            all_names.update(chunk['name'].unique())
            all_main_categories.update(chunk['main_category'].unique())
            all_sub_categories.update(chunk['sub_category'].unique())
        
        print(f"Found {len(all_names)} unique products")
        print(f"Found {len(all_main_categories)} unique main categories")
        print(f"Found {len(all_sub_categories)} unique sub categories")
        
        # 2. Create and fit encoders with all unique values
        name_enc = LabelEncoder()
        main_cat_enc = LabelEncoder()
        sub_cat_enc = LabelEncoder()
        
        name_enc.fit(list(all_names))
        main_cat_enc.fit(list(all_main_categories))
        sub_cat_enc.fit(list(all_sub_categories))
        
        # Create lookup dictionaries
        id_to_name = dict(zip(range(len(all_names)), name_enc.classes_))
        name_to_id = {name: idx for idx, name in id_to_name.items()}
        num_products = len(all_names)
        
        # 3. Preallocate product_meta array
        product_meta = np.zeros((num_products, 2), dtype=np.int32)
        
        # 4. Second pass: Process data for product_meta
        usecols = ["name", "main_category", "sub_category"]
        for chunk in pd.read_csv(csv_path, usecols=usecols, chunksize=chunk_size):
            for _, row in chunk.iterrows():
                try:
                    pid = name_to_id[row['name']]
                    main_cat = main_cat_enc.transform([row['main_category']])[0]
                    sub_cat = sub_cat_enc.transform([row['sub_category']])[0]
                    product_meta[pid, 0] = main_cat
                    product_meta[pid, 1] = sub_cat
                except KeyError as e:
                    print(f"Warning: Could not process row with name {row['name']}: {e}")
        
        print(f"Memory usage after creating metadata: {get_memory_usage_mb():.2f} MB")
        
        # 5. Load TFLite model
        interpreter = tf.lite.Interpreter(model_path=model_path)
        interpreter.allocate_tensors()
        print("TFLite model loaded.")
        
        # Extract embeddings
        def get_embedding_weights(layer_name):
            for tensor_detail in interpreter.get_tensor_details():
                if layer_name in tensor_detail["name"]:
                    tensor = interpreter.get_tensor(tensor_detail["index"])
                    print(f"Found tensor: {tensor_detail['name']} with shape {tensor.shape}")
                    if tensor.ndim >= 2:
                        return tensor.astype(np.float32)
            raise ValueError(f"Tensor for {layer_name} with at least 2 dimensions not found")

        # Extract embeddings
        prod_weights = get_embedding_weights("product_embedding")
        main_weights = get_embedding_weights("main_cat_embedding")
        sub_weights = get_embedding_weights("sub_cat_embedding")
        
        # Free resources
        del interpreter
        interpreter = None
        gc.collect()
        
        print(f"Final memory usage after setup: {get_memory_usage_mb():.2f} MB")
        print("Startup completed. API is ready.")

    except Exception as e:
        print(f"Error during startup: {str(e)}")
        raise e

    yield
    print("Shutting down API...")

# Create FastAPI application
app = FastAPI(lifespan=lifespan, title="Product Recommendation API")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Health Check
@app.get("/")
def health_check():
    return {"status": "ok"}

# Functions to get embeddings
def get_combined_embedding(product_id: int):
    main_cat = product_meta[product_id, 0]
    sub_cat = product_meta[product_id, 1]
    vec_prod = prod_weights[product_id]
    vec_main = main_weights[main_cat]
    vec_sub = sub_weights[sub_cat]
    return np.concatenate([vec_prod, vec_main, vec_sub])

def recomendar_productos_combinados(product_id: int, top_n: int = 5):
    if product_id < 0 or product_id >= num_products:
        return []
    
    query_vec = get_combined_embedding(product_id).reshape(1, -1)
    
    # Memory optimization: Process in batches
    batch_size = 100
    similitudes = np.zeros(num_products, dtype=np.float32)
    
    for start_idx in range(0, num_products, batch_size):
        end_idx = min(start_idx + batch_size, num_products)
        batch_ids = list(range(start_idx, end_idx))
        
        # Get embeddings for this batch
        batch_embeddings = np.vstack([get_combined_embedding(pid) for pid in batch_ids])
        
        # Calculate similarities for this batch
        batch_similarities = cosine_similarity(query_vec, batch_embeddings)[0]
        similitudes[start_idx:end_idx] = batch_similarities
    
    # Get top-n similar products
    indices_similares = np.argsort(-similitudes)
    indices_similares = [i for i in indices_similares if i != product_id]
    return indices_similares[:top_n]

# Recommendation endpoint
@app.get("/recomendaciones")
def get_recomendaciones(product_name: str, top_n: int = 5):
    if product_name not in name_to_id:
        raise HTTPException(status_code=404, detail="Product not found")
    
    product_id = name_to_id[product_name]
    recomendados_ids = recomendar_productos_combinados(product_id, top_n)
    
    recomendaciones = [
        {"id": int(pid), "name": id_to_name.get(int(pid), f"ID {pid}")}
        for pid in recomendados_ids
    ]
    
    return {"producto": product_name, "recomendaciones": recomendaciones}

# Run with Uvicorn if executed directly
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=port)
