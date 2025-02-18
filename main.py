import os
import gc
import psutil
import numpy as np
import pandas as pd
from typing import Dict, List
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from sklearn.preprocessing import LabelEncoder
import logging
import tempfile
from pathlib import Path
# Add this import at the top with the other imports
from typing import Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Create FastAPI app - mantener exactamente la misma estructura
app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    global df, interpreter, id_to_name, name_to_id
    global prod_weights, main_weights, sub_weights, num_products, product_meta

    print("\n=== INICIO DEL PROCESO DE CARGA ===")
    try:
        # Verificar rutas de archivos
        csv_path = os.path.join(BASE_DIR, "productos.csv")
        model_path = os.path.join(BASE_DIR, "recomendacion.tflite")
        
        print(f"\n1. Verificación de archivos:")
        print(f"   - BASE_DIR: {BASE_DIR}")
        print(f"   - CSV Path: {csv_path}")
        print(f"   - ¿Existe CSV?: {os.path.exists(csv_path)}")
        
        if not os.path.exists(csv_path):
            print("   - Contenido del directorio:")
            for file in os.listdir(BASE_DIR):
                print(f"     * {file}")
            raise FileNotFoundError(f"No se encontró productos.csv en {csv_path}")

        # 1. Cargar CSV con debug
        print("\n2. Cargando CSV:")
        usecols = ["name", "discount_price", "actual_price", "main_category", "sub_category"]
        print(f"   - Columnas a cargar: {usecols}")
        
        try:
            # Primero leer solo las primeras líneas para verificar
            print("   - Verificando primeras líneas del CSV:")
            df_preview = pd.read_csv(csv_path, nrows=5)
            print("   - Columnas encontradas en CSV:", df_preview.columns.tolist())
            
            # Cargar el CSV completo
            df = pd.read_csv(csv_path, usecols=usecols, dtype=str)
            print(f"   - CSV cargado exitosamente con {len(df)} filas")
            print(f"   - Muestra de datos:\n{df.head()}")
            
            # Verificar valores nulos
            null_counts = df.isnull().sum()
            print(f"   - Valores nulos por columna:\n{null_counts}")
            
        except Exception as e:
            print(f"   - Error al cargar CSV: {str(e)}")
            raise

        # Debug de procesamiento de precios
        print("\n3. Procesamiento de precios:")
        try:
            print("   - Valores únicos en discount_price antes de limpieza:", df['discount_price'].unique()[:5])
            df['discount_price'] = df['discount_price'].str.replace('₹', '').str.replace(',', '')
            print("   - Valores después de limpieza:", df['discount_price'].unique()[:5])
            df['discount_price'] = df['discount_price'].astype(np.float32)
            print("   - Valores después de conversión a float:", df['discount_price'].unique()[:5])
            
            # Repetir para actual_price
            print("\n   - Procesando actual_price...")
            df['actual_price'] = df['actual_price'].str.replace('₹', '').str.replace(',', '').astype(np.float32)
        except Exception as e:
            print(f"   - Error en procesamiento de precios: {str(e)}")
            raise

        # Debug de codificación de variables
        print("\n4. Codificación de variables:")
        try:
            name_enc = LabelEncoder()
            print("   - Nombres únicos antes de codificación:", len(df['name'].unique()))
            df['name_encoded'] = name_enc.fit_transform(df['name'])
            print("   - Nombres únicos después de codificación:", len(df['name_encoded'].unique()))
            print("   - Muestra de codificación:")
            sample_encoding = df[['name', 'name_encoded']].head()
            print(sample_encoding)
            
            # Crear y verificar mapeos
            id_to_name = dict(zip(df['name_encoded'], df['name']))
            name_to_id = {v: k for k, v in id_to_name.items()}
            
            print(f"\n5. Verificación de mapeos:")
            print(f"   - Tamaño de id_to_name: {len(id_to_name)}")
            print(f"   - Tamaño de name_to_id: {len(name_to_id)}")
            print(f"   - Primeros 5 elementos de name_to_id:")
            for i, (k, v) in enumerate(list(name_to_id.items())[:5]):
                print(f"     {k}: {v}")
            
            # Verificar integridad de los mapeos
            print("\n6. Verificación de integridad:")
            print(f"   - ¿Todos los IDs tienen nombre?: {all(i in id_to_name for i in range(len(id_to_name)))}")
            print(f"   - ¿Todos los nombres tienen ID?: {all(name in name_to_id for name in df['name'])}")
            
            # Verificar duplicados
            duplicates = df['name'].value_counts()[df['name'].value_counts() > 1]
            if not duplicates.empty:
                print("\n   - Nombres duplicados encontrados:")
                print(duplicates)
            
        except Exception as e:
            print(f"   - Error en codificación de variables: {str(e)}")
            raise

        print("\n=== FIN DEL PROCESO DE CARGA ===")

    except Exception as e:
        print(f"\nERROR FATAL: {str(e)}")
        import traceback
        print(f"Traceback completo:\n{traceback.format_exc()}")
        raise e

    yield
    print("Apagando API...")


@app.on_event("startup")
async def startup_event():
    port = os.environ.get("PORT", 10000)
    logger.info(f"Application startup on port {port}")

class MemoryEfficientStore:
    def __init__(self, chunk_size: int = 100):
        self.chunk_size = chunk_size
        self.temp_dir = Path(tempfile.mkdtemp())
        self.chunks = []
        self.current_chunk = {}
        
    def add_embedding(self, idx: int, embedding: np.ndarray):
        self.current_chunk[idx] = embedding.astype(np.float16)
        
        if len(self.current_chunk) >= self.chunk_size:
            self._save_chunk()
    
    def _save_chunk(self):
        if not self.current_chunk:
            return
            
        chunk_file = self.temp_dir / f"chunk_{len(self.chunks)}.npz"
        np.savez_compressed(
            chunk_file,
            indices=np.array(list(self.current_chunk.keys())),
            embeddings=np.stack(list(self.current_chunk.values()))
        )
        self.chunks.append(chunk_file)
        self.current_chunk.clear()
        gc.collect()
    
    def get_similarities(self, query_id: int) -> Dict[int, float]:
        query_embedding = None
        
        # Find query embedding
        for chunk_file in self.chunks:
            data = np.load(chunk_file)
            idx = np.where(data['indices'] == query_id)[0]
            if len(idx) > 0:
                query_embedding = data['embeddings'][idx[0]]
                break
        
        if query_embedding is None and query_id in self.current_chunk:
            query_embedding = self.current_chunk[query_id]
        
        if query_embedding is None:
            raise ValueError(f"Query ID {query_id} not found")
        
        # Calculate similarities
        similarities = {}
        for chunk_file in self.chunks:
            data = np.load(chunk_file)
            chunk_sims = np.dot(data['embeddings'], query_embedding)
            for idx, sim in zip(data['indices'], chunk_sims):
                if idx != query_id:
                    similarities[int(idx)] = float(sim)
        
        # Process current chunk
        for idx, emb in self.current_chunk.items():
            if idx != query_id:
                similarities[idx] = float(np.dot(emb, query_embedding))
        
        return similarities
    
    def cleanup(self):
        for chunk_file in self.chunks:
            try:
                chunk_file.unlink()
            except Exception as e:
                logger.error(f"Error removing {chunk_file}: {e}")
        try:
            self.temp_dir.rmdir()
        except Exception as e:
            logger.error(f"Error removing temp dir: {e}")

class RecommendationSystem:
    def __init__(self):
        self.id_to_name: Dict[int, str] = {}
        self.name_to_id: Dict[str, int] = {}
        self.embedding_store = MemoryEfficientStore()
    
    def process_data(self, csv_path: str):
        chunk_size = 500
        
        # Process unique values
        unique_values = {"name": set(), "category": set()}
        for chunk in pd.read_csv(csv_path, usecols=list(unique_values.keys()), 
                               dtype=str, chunksize=chunk_size):
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
        self.id_to_name = dict(enumerate(encoders["name"].classes_))
        self.name_to_id = {name: idx for idx, name in self.id_to_name.items()}
        
        # Process data in chunks
        for chunk in pd.read_csv(csv_path, chunksize=chunk_size, dtype=str):
            valid_mask = chunk['name'].isin(self.name_to_id)
            if not valid_mask.any():
                continue
            
            chunk = chunk[valid_mask]
            for _, row in chunk.iterrows():
                product_id = self.name_to_id[row['name']]
                
                # Create simple embedding from category
                category_id = encoders["category"].transform([row['category']])[0]
                embedding = np.zeros(50, dtype=np.float32)
                embedding[category_id % 50] = 1
                
                # Store embedding
                self.embedding_store.add_embedding(product_id, embedding)
            
            del chunk
            gc.collect()
        
        # Save final chunk
        self.embedding_store._save_chunk()
    
    def get_recommendations(self, product_name: str, top_n: int = 5) -> List[Dict]:
        if product_name not in self.name_to_id:
            raise HTTPException(status_code=404, detail="Product not found")
        
        product_id = self.name_to_id[product_name]
        try:
            similarities = self.embedding_store.get_similarities(product_id)
            
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

@app.on_event("startup")
async def startup_event():
    port = os.environ.get("PORT", 10000)
    logger.info(f"Application startup on port {port}")
    
    try:
        # Get file path
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        csv_path = os.path.join(BASE_DIR, "productos.csv")
        
        # Process data
        logger.info("Starting data processing")
        recommender.process_data(csv_path)
        logger.info("Data processing complete")
        
    except Exception as e:
        logger.error(f"Error during startup: {e}")
        # No raise here - let the app start even if data processing fails

@app.on_event("shutdown")
async def shutdown_event():
    logger.info("Shutting down")
    recommender.embedding_store.cleanup()

@app.get("/")
async def root():
    memory_mb = psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)
    return {
        "status": "ok",
        "memory_usage_mb": memory_mb
    }

@app.get("/health")
async def health_check():
    memory_mb = psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)
    return {
        "status": "healthy",
        "memory_usage_mb": memory_mb,
        "num_products": len(recommender.id_to_name)
    }

@app.get("/recomendaciones")
async def get_recommendations(product_name: str, top_n: int = 5):
    return {
        "producto": product_name,
        "recomendaciones": recommender.get_recommendations(product_name, top_n)
    }

# Add this new endpoint after the existing endpoints
@app.get("/productos")
async def list_products(
    page: Optional[int] = 1,
    page_size: Optional[int] = 50,
    search: Optional[str] = None
):
    """
    List available products with optional pagination and search.
    
    Args:
        page: Current page number (1-based indexing)
        page_size: Number of items per page
        search: Optional search term to filter products
    """
    # Validate pagination parameters
    if page < 1:
        raise HTTPException(status_code=400, detail="Page number must be >= 1")
    if page_size < 1 or page_size > 100:
        raise HTTPException(status_code=400, detail="Page size must be between 1 and 100")
    
    # Get all product names
    products = list(recommender.name_to_id.keys())
    
    # Apply search filter if provided
    if search:
        search = search.lower()
        products = [p for p in products if search in p.lower()]
    
    # Sort products alphabetically for consistent pagination
    products.sort()
    
    # Calculate pagination
    total_products = len(products)
    total_pages = (total_products + page_size - 1) // page_size
    
    # Adjust page if it exceeds total pages
    if page > total_pages and total_pages > 0:
        raise HTTPException(status_code=404, detail="Page not found")
    
    # Get paginated results
    start_idx = (page - 1) * page_size
    end_idx = min(start_idx + page_size, total_products)
    page_products = products[start_idx:end_idx]
    
    return {
        "productos": page_products,
        "pagina_actual": page,
        "total_paginas": total_pages,
        "productos_por_pagina": page_size,
        "total_productos": total_products
    }


@app.get("/debug")
async def debug_info():
    """Endpoint para diagnóstico del sistema"""
    try:
        return {
            "status": "ok",
            "memory_usage_mb": get_memory_usage_mb(),
            "system_state": {
                "df_loaded": df is not None,
                "df_shape": df.shape if df is not None else None,
                "df_columns": df.columns.tolist() if df is not None else None,
                "name_to_id_size": len(name_to_id) if name_to_id is not None else 0,
                "id_to_name_size": len(id_to_name) if id_to_name is not None else 0,
                "num_products": num_products,
                "sample_names": list(name_to_id.keys())[:5] if name_to_id is not None else [],
                "product_meta_shape": product_meta.shape if product_meta is not None else None,
            },
            "file_info": {
                "base_dir": BASE_DIR,
                "csv_exists": os.path.exists(os.path.join(BASE_DIR, "productos.csv")),
                "model_exists": os.path.exists(os.path.join(BASE_DIR, "recomendacion.tflite")),
                "directory_contents": os.listdir(BASE_DIR)
            }
        }
    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
            "traceback": traceback.format_exc()
        }
