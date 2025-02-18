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

# Función para obtener uso de memoria
def get_memory_usage_mb():
    process = psutil.Process(os.getpid())
    mem_bytes = process.memory_info().rss
    return mem_bytes / (1024 * 1024)

print(f"Uso de memoria al inicio: {get_memory_usage_mb():.2f} MB")

# Variables globales
df = None
interpreter = None
id_to_name = None
name_to_id = None
prod_weights = None
main_weights = None
sub_weights = None
num_products = None
product_meta = None

# Definir el directorio de trabajo
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

@asynccontextmanager
async def lifespan(app: FastAPI):
    global df, interpreter, id_to_name, name_to_id
    global prod_weights, main_weights, sub_weights, num_products, product_meta

    print("Iniciando startup...")
    try:
        # Ruta de los archivos
        csv_path = os.path.join(BASE_DIR, "productos.csv")
        model_path = os.path.join(BASE_DIR, "recomendacion.tflite")

        # 1. Cargar CSV
        usecols = ["name", "discount_price", "actual_price", "main_category", "sub_category"]
        df = pd.read_csv(csv_path, usecols=usecols, dtype=str)
        print(f"Uso de memoria después de cargar CSV: {get_memory_usage_mb():.2f} MB")

        # Convertir precios y normalizarlos
        df['discount_price'] = df['discount_price'].str.replace('₹', '').str.replace(',', '').astype(np.float32)
        df['actual_price'] = df['actual_price'].str.replace('₹', '').str.replace(',', '').astype(np.float32)
        df['discount_price'] /= df['discount_price'].max()
        df['actual_price'] /= df['actual_price'].max()

        print("Preprocesamiento de precios completado.")

        # Codificar variables
        name_enc = LabelEncoder()
        df['name_encoded'] = name_enc.fit_transform(df['name'])
        main_cat_enc = LabelEncoder()
        df['main_category_encoded'] = main_cat_enc.fit_transform(df['main_category'])
        sub_cat_enc = LabelEncoder()
        df['sub_category_encoded'] = sub_cat_enc.fit_transform(df['sub_category'])

        print("Variables codificadas.")

        # Mapeos de nombres e IDs
        id_to_name = dict(zip(df['name_encoded'], df['name']))
        name_to_id = {v: k for k, v in id_to_name.items()}

        print("Mapeos creados.")

        # Crear matriz de metadatos de productos (main_category y sub_category)
        num_products = df['name_encoded'].max() + 1
        product_meta = np.zeros((num_products, 2), dtype=np.int32)
        for _, row in df.iterrows():
            pid = row['name_encoded']
            product_meta[pid, 0] = row['main_category_encoded']
            product_meta[pid, 1] = row['sub_category_encoded']

        # 2. Cargar modelo TFLite
        interpreter = tf.lite.Interpreter(model_path=model_path)
        interpreter.allocate_tensors()
        print("Modelo TFLite cargado.")

        # Función para extraer pesos de embeddings
        def get_embedding_weights(layer_name):
            encontrado = False
            tensor_final = None
            for tensor_detail in interpreter.get_tensor_details():
                if layer_name in tensor_detail["name"]:
                    tensor = interpreter.get_tensor(tensor_detail["index"])
                    print(f"Encontrado tensor: {tensor_detail['name']} con shape {tensor.shape}")
                    # Se espera que el tensor tenga al menos 2 dimensiones
                    if tensor.ndim >= 2:
                        tensor_final = tensor.astype(np.float32)
                        encontrado = True
                        break
            if not encontrado:
                raise ValueError(f"No se encontró tensor para {layer_name} con al menos 2 dimensiones")
            return tensor_final

        # Extraer embeddings
        prod_weights = get_embedding_weights("product_embedding")
        main_weights = get_embedding_weights("main_cat_embedding")
        sub_weights = get_embedding_weights("sub_cat_embedding")

        print("Pesos de embeddings extraídos.")

        # Liberar recursos del intérprete
        del interpreter
        interpreter = None
        gc.collect()

        print("Startup completado. La API está lista.")

    except Exception as e:
        print("Error en startup:", e)
        raise e

    yield
    print("Apagando API...")

# Crear la aplicación FastAPI
app = FastAPI(lifespan=lifespan, title="API de Recomendaciones de Productos")

# Middleware CORS
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

# Funciones para obtener embeddings combinados
def get_combined_embedding(product_id: int):
    main_cat = product_meta[product_id, 0]
    sub_cat = product_meta[product_id, 1]
    vec_prod = prod_weights[product_id]      # Se espera que prod_weights sea de shape (num_products, embedding_dim)
    vec_main = main_weights[main_cat]
    vec_sub = sub_weights[sub_cat]
    return np.concatenate([vec_prod, vec_main, vec_sub]).astype(np.float32)

def get_all_combined_embeddings():
    return np.concatenate([
        prod_weights,
        main_weights[product_meta[:, 0]],
        sub_weights[product_meta[:, 1]]
    ], axis=1).astype(np.float32)

def recomendar_productos_combinados(product_id: int, top_n: int = 5):
    if product_id < 0 or product_id >= num_products:
        return []
    query_vec = get_combined_embedding(product_id).reshape(1, -1)
    combined_embeddings = get_all_combined_embeddings()
    similitudes = cosine_similarity(query_vec, combined_embeddings)[0]
    indices_similares = np.argsort(-similitudes)
    # Excluir el producto consultado
    indices_similares = [i for i in indices_similares if i != product_id]
    return indices_similares[:top_n]

# Endpoint de Recomendaciones
@app.get("/recomendaciones")
def get_recomendaciones(product_name: str, top_n: int = 5):
    if product_name not in name_to_id:
        raise HTTPException(status_code=404, detail="Producto no encontrado")
    product_id = name_to_id[product_name]
    recomendados_ids = recomendar_productos_combinados(product_id, top_n)
    recomendaciones = [
        {"id": int(pid), "name": id_to_name.get(int(pid), f"ID {pid}")}
        for pid in recomendados_ids
    ]
    return {"producto": product_name, "recomendaciones": recomendaciones}

# Ejecutar Uvicorn si se corre directamente
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=port)
