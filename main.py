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

# Memory optimization: Use memory-mapped files
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MEM_MAP_DIR = os.path.join(BASE_DIR, "mem_maps")
os.makedirs(MEM_MAP_DIR, exist_ok=True)

# Function to get memory usage
def get_memory_usage_mb():
    process = psutil.Process(os.getpid())
    mem_bytes = process.memory_info().rss
    return mem_bytes / (1024 * 1024)

print(f"Initial memory usage: {get_memory_usage_mb():.2f} MB")

# Global variables (minimized)
id_to_name = None
name_to_id = None
num_products = None
product_meta = None

# Memory-mapped embedding arrays
prod_weights_mmap = None
main_weights_mmap = None
sub_weights_mmap = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global id_to_name, name_to_id, num_products, product_meta
    global prod_weights_mmap, main_weights_mmap, sub_weights_mmap

    print("Starting startup...")
    try:
        # File paths
        csv_path = os.path.join(BASE_DIR, "productos.csv")
        model_path = os.path.join(BASE_DIR, "recomendacion.tflite")

        # Memory optimization: Load only essential columns and process in chunks
        usecols = ["name", "discount_price", "actual_price", "main_category", "sub_category"]
        
        # First pass to count rows
        num_rows = sum(1 for _ in open(csv_path)) - 1
        
        # Initialize encoders
        name_enc = LabelEncoder()
        main_cat_enc = LabelEncoder()
        sub_cat_enc = LabelEncoder()
        
        # Process the file in chunks to fit all unique values
        chunk_size = 1000
        for chunk in pd.read_csv(csv_path, usecols=usecols, chunksize=chunk_size):
            name_enc.fit(chunk['name'])
            main_cat_enc.fit(chunk['main_category'])
            sub_cat_enc.fit(chunk['sub_category'])
        
        # Create dictionaries for lookups
        unique_names = name_enc.classes_
        id_to_name = {i: name for i, name in enumerate(unique_names)}
        name_to_id = {name: i for i, name in enumerate(unique_names)}
        
        num_products = len(unique_names)
        print(f"Total unique products: {num_products}")
        
        # Memory optimization: Pre-allocate product_meta array as memory-mapped file
        product_meta_path = os.path.join(MEM_MAP_DIR, "product_meta.npy")
        product_meta = np.memmap(product_meta_path, dtype=np.int16, mode='w+', shape=(num_products, 2))
        
        # Process in chunks to populate product_meta
        curr_idx = 0
        for chunk in pd.read_csv(csv_path, usecols=usecols, chunksize=chunk_size):
            chunk_size = len(chunk)
            for _, row in chunk.iterrows():
                pid = name_enc.transform([row['name']])[0]
                main_cat = main_cat_enc.transform([row['main_category']])[0]
                sub_cat = sub_cat_enc.transform([row['sub_category']])[0]
                product_meta[pid, 0] = main_cat
                product_meta[pid, 1] = sub_cat
            
            curr_idx += chunk_size
            print(f"Processed {curr_idx}/{num_rows} rows")
        
        print(f"Memory usage after creating metadata: {get_memory_usage_mb():.2f} MB")
        
        # Load TFLite model
        interpreter = tf.lite.Interpreter(model_path=model_path)
        interpreter.allocate_tensors()
        print("TFLite model loaded.")
        
        # Function to extract embedding weights and save as memory-mapped file
        def get_embedding_weights(layer_name, shape_hint=None):
            for tensor_detail in interpreter.get_tensor_details():
                if layer_name in tensor_detail["name"]:
                    tensor = interpreter.get_tensor(tensor_detail["index"])
                    print(f"Found tensor: {tensor_detail['name']} with shape {tensor.shape}")
                    if tensor.ndim >= 2:
                        # Save to memory-mapped file
                        output_path = os.path.join(MEM_MAP_DIR, f"{layer_name}.npy")
                        tensor_mmap = np.memmap(output_path, dtype=np.float16, mode='w+', shape=tensor.shape)
                        tensor_mmap[:] = tensor.astype(np.float16)
                        tensor_mmap.flush()
                        return tensor_mmap
            
            raise ValueError(f"Tensor for {layer_name} with at least 2 dimensions not found")

        # Extract embeddings as memory-mapped files
        prod_weights_mmap = get_embedding_weights("product_embedding")
        main_weights_mmap = get_embedding_weights("main_cat_embedding")
        sub_weights_mmap = get_embedding_weights("sub_cat_embedding")
        
        # Free resources
        del interpreter
        interpreter = None
        gc.collect()
        
        print(f"Final memory usage after setup: {get_memory_usage_mb():.2f} MB")
        print("Startup completed. API is ready.")

    except Exception as e:
        print("Error during startup:", e)
        raise e

    yield
    print("Shutting down API...")
    
    # Clean up memory-mapped files
    if prod_weights_mmap is not None:
        del prod_weights_mmap
    if main_weights_mmap is not None:
        del main_weights_mmap
    if sub_weights_mmap is not None:
        del sub_weights_mmap
    if product_meta is not None:
        del product_meta
    gc.collect()

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

# Optimized functions to get embeddings
def get_combined_embedding(product_id: int):
    main_cat = product_meta[product_id, 0]
    sub_cat = product_meta[product_id, 1]
    vec_prod = prod_weights_mmap[product_id].astype(np.float16)
    vec_main = main_weights_mmap[main_cat].astype(np.float16)
    vec_sub = sub_weights_mmap[sub_cat].astype(np.float16)
    return np.concatenate([vec_prod, vec_main, vec_sub])

# Optimized recommendation function
def recomendar_productos_combinados(product_id: int, top_n: int = 5):
    if product_id < 0 or product_id >= num_products:
        return []
    
    query_vec = get_combined_embedding(product_id).reshape(1, -1)
    
    # Memory optimization: Process in batches instead of all at once
    batch_size = 100
    similitudes = np.zeros(num_products, dtype=np.float16)
    
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
