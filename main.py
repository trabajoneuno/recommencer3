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

def get_memory_usage_mb():
    process = psutil.Process(os.getpid())
    mem_bytes = process.memory_info().rss
    return mem_bytes / (1024 * 1024)

print(f"Initial memory usage: {get_memory_usage_mb():.2f} MB")

# Variables globales
id_to_name = None
name_to_id = None
num_products = None
product_meta = None
prod_weights = None
main_weights = None
sub_weights = None
combined_embeddings_norm = None

# Directorio base (donde se encuentra el CSV y el modelo)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

@asynccontextmanager
async def lifespan(app: FastAPI):
    global id_to_name, name_to_id, num_products, product_meta
    global prod_weights, main_weights, sub_weights, combined_embeddings_norm

    print("Starting startup...")
    try:
        # Rutas a archivos
        csv_path = os.path.join(BASE_DIR, "productos.csv")
        model_path = os.path.join(BASE_DIR, "recomendacion.tflite")
        
        # === Primera pasada: recopilar valores únicos ===
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
        
        # === Ajustar codificadores ===
        name_enc = LabelEncoder()
        main_cat_enc = LabelEncoder()
        sub_cat_enc = LabelEncoder()
        
        name_enc.fit(list(all_names))
        main_cat_enc.fit(list(all_main_categories))
        sub_cat_enc.fit(list(all_sub_categories))
        
        # Crear diccionarios de mapeo para productos
        id_to_name = dict(zip(range(len(name_enc.classes_)), name_enc.classes_))
        name_to_id = {name: idx for idx, name in id_to_name.items()}
        num_products = len(name_enc.classes_)
        
        # Preasignar array para metadata de productos:
        # Columna 0: código de main_category, Columna 1: código de sub_category
        product_meta = np.zeros((num_products, 2), dtype=np.int32)
        
        # === Segunda pasada: procesar CSV de forma vectorizada ===
        for chunk in pd.read_csv(csv_path, usecols=usecols, chunksize=chunk_size, dtype=str):
            # Mapear nombres a sus IDs
            ids = chunk['name'].map(name_to_id)
            # Omitir registros sin mapeo (por precaución)
            if ids.isnull().any():
                valid = ids.notnull()
                chunk = chunk[valid]
                ids = ids[valid]
            ids = ids.astype(np.int32).values
            # Transformar categorías de forma vectorizada
            main_encoded = main_cat_enc.transform(chunk['main_category'].values)
            sub_encoded = sub_cat_enc.transform(chunk['sub_category'].values)
            # Asignar en el array preasignado
            product_meta[ids, 0] = main_encoded
            product_meta[ids, 1] = sub_encoded
        
        print(f"Memory usage after creating metadata: {get_memory_usage_mb():.2f} MB")
        
        # === Cargar modelo TFLite ===
        interpreter = tf.lite.Interpreter(model_path=model_path)
        interpreter.allocate_tensors()
        print("TFLite model loaded.")
        
        def get_embedding_weights(layer_name):
            for tensor_detail in interpreter.get_tensor_details():
                if layer_name in tensor_detail["name"]:
                    tensor = interpreter.get_tensor(tensor_detail["index"])
                    print(f"Found tensor: {tensor_detail['name']} with shape {tensor.shape}")
                    if tensor.ndim >= 2:
                        return tensor.astype(np.float32)
            raise ValueError(f"Tensor for {layer_name} with at least 2 dimensions not found")
        
        prod_weights = get_embedding_weights("product_embedding")
        main_weights = get_embedding_weights("main_cat_embedding")
        sub_weights = get_embedding_weights("sub_cat_embedding")
        
        # Liberar recursos del intérprete
        del interpreter
        gc.collect()
        
        # === Precalcular embeddings combinados normalizados ===
        # Se asume:
        # - prod_weights: (num_products, d_prod)
        # - main_weights: (num_main_categories, d_main)
        # - sub_weights: (num_sub_categories, d_sub)
        # Se obtiene combined_embeddings de forma vectorizada:
        combined_embeddings = np.concatenate([
            prod_weights,
            main_weights[product_meta[:, 0]],
            sub_weights[product_meta[:, 1]]
        ], axis=1)
        
        # Normalizar (añadimos un epsilon para evitar división por cero)
        norms = np.linalg.norm(combined_embeddings, axis=1, keepdims=True)
        combined_embeddings_norm = combined_embeddings / (norms + 1e-10)
        
        print(f"Final memory usage after setup: {get_memory_usage_mb():.2f} MB")
        print("Startup completed. API is ready.")
    
    except Exception as e:
        print(f"Error during startup: {str(e)}")
        raise e
    
    yield
    print("Shutting down API...")

# Crear aplicación FastAPI y configurar CORS
app = FastAPI(lifespan=lifespan, title="Product Recommendation API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Endpoint de Health Check
@app.get("/")
def health_check():
    return {"status": "ok"}

# Endpoint de recomendaciones
@app.get("/recomendaciones")
def get_recomendaciones(product_name: str, top_n: int = 5):
    global id_to_name, name_to_id, combined_embeddings_norm
    
    if product_name not in name_to_id:
        raise HTTPException(status_code=404, detail="Producto no encontrado")
    
    product_id = name_to_id[product_name]
    # Obtener embedding del producto (ya normalizado)
    query_embedding = combined_embeddings_norm[product_id]
    # Calcular similitud coseno de forma vectorizada (producto punto)
    similarities = np.dot(combined_embeddings_norm, query_embedding)
    # Excluir el propio producto
    similarities[product_id] = -np.inf
    
    # Obtener los índices de los top_n productos recomendados
    top_indices = np.argpartition(-similarities, top_n)[:top_n]
    # Ordenarlos de forma descendente según la similitud
    top_indices = top_indices[np.argsort(-similarities[top_indices])]
    
    recomendaciones = [
        {"id": int(pid), "name": id_to_name.get(int(pid), f"ID {pid}")}
        for pid in top_indices
    ]
    
    return {"producto": product_name, "recomendaciones": recomendaciones}

# Ejecutar la aplicación con Uvicorn si se ejecuta directamente
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=port)
