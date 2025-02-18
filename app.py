from flask import Flask, request, jsonify
import pickle
import pandas as pd
from pathlib import Path
import os
import logging
from recommender import ImprovedRecommender

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Variables globales para el modelo y datos
recommender = None
products_df = None
interactions_df = None

class CustomUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        # Si el pickle busca __main__.ImprovedRecommender, redirige a recommender.ImprovedRecommender
        if module == "__main__" and name == "ImprovedRecommender":
            module = "recommender"
        return super().find_class(module, name)

def init_app():
    """Inicializar la aplicación y cargar los modelos"""
    global recommender, products_df, interactions_df
    
    try:
        logger.info("Iniciando carga de modelos...")
        models_dir = Path("models")
        
        # Primero cargamos los DataFrames
        logger.info("Cargando DataFrames...")
        products_df = pd.read_pickle(models_dir / 'products_df.pkl')
        interactions_df = pd.read_pickle(models_dir / 'interactions_df.pkl')
        
        # Luego cargamos el recomendador, usando CustomUnpickler para redirigir correctamente
        logger.info("Cargando recomendador...")
        with open(models_dir / 'recommender.pkl', 'rb') as f:
            recommender = CustomUnpickler(f).load()
            
        logger.info("Inicializando directorio de datos...")
        # Nos aseguramos que el directorio de datos exista
        recommender.data_dir = Path('recommender_data')
        recommender.data_dir.mkdir(exist_ok=True)
        
        logger.info("Modelos cargados exitosamente")
        return True
    except Exception as e:
        logger.error(f"Error loading models: {str(e)}")
        return False

# Inicializar la aplicación Flask
app = Flask(__name__)

# Intentar cargar los modelos al inicio
init_success = init_app()

@app.route('/', methods=['GET'])
def home():
    return jsonify({
        "health_check": "OK",
        "models_loaded": [
            "recommender_system"
        ]
    })

@app.route('/health', methods=['GET'])
def health_check():
    """Endpoint para verificar que la API está funcionando"""
    global init_success, recommender, products_df, interactions_df
    
    if not init_success:
        init_success = init_app()
        
    return jsonify({
        "status": "healthy" if init_success else "error",
        "recommender_loaded": recommender is not None,
        "products_loaded": products_df is not None,
        "interactions_loaded": interactions_df is not None
    }), 200 if init_success else 500

@app.route('/recommend/user/<int:user_id>', methods=['GET'])
def get_user_recommendations(user_id):
    """Obtener recomendaciones para un usuario específico"""
    global init_success, recommender
    
    if not init_success:
        init_success = init_app()
    
    if not init_success:
        return jsonify({"error": "Recommender system not initialized"}), 500
        
    try:
        n_recommendations = int(request.args.get('n', 5))
        logger.info(f"Getting recommendations for user {user_id}")
        recommendations = recommender.get_recommendations(user_id, n_recommendations)
        return jsonify({
            "user_id": user_id,
            "recommendations": recommendations
        })
    except Exception as e:
        logger.error(f"Error getting recommendations for user {user_id}: {str(e)}")
        return jsonify({"error": str(e)}), 400

@app.route('/recommend/history', methods=['POST'])
def get_history_recommendations():
    """Obtener recomendaciones basadas en historial de productos"""
    global init_success, recommender
    
    if not init_success:
        init_success = init_app()
    
    if not init_success:
        return jsonify({"error": "Recommender system not initialized"}), 500
        
    try:
        data = request.get_json()
        if not data or 'product_ids' not in data:
            return jsonify({"error": "product_ids list is required"}), 400
            
        product_ids = data['product_ids']
        n_recommendations = data.get('n', 5)
        
        logger.info(f"Getting recommendations from product history: {product_ids}")
        recommendations = recommender.get_recommendations_from_history(product_ids, n_recommendations)
        return jsonify({
            "product_history": product_ids,
            "recommendations": recommendations
        })
    except Exception as e:
        logger.error(f"Error getting recommendations from history: {str(e)}")
        return jsonify({"error": str(e)}), 400

@app.route('/recommend/popular', methods=['GET'])
def get_popular_recommendations():
    """Obtener recomendaciones populares"""
    global init_success, recommender
    
    if not init_success:
        init_success = init_app()
    
    if not init_success:
        return jsonify({"error": "Recommender system not initialized"}), 500
        
    try:
        n_recommendations = int(request.args.get('n', 5))
        logger.info("Getting popular recommendations")
        recommendations = recommender._get_popular_recommendations(n_recommendations)
        return jsonify({
            "recommendations": recommendations
        })
    except Exception as e:
        logger.error(f"Error getting popular recommendations: {str(e)}")
        return jsonify({"error": str(e)}), 400

@app.route('/recommend', methods=['POST'])
def get_recommendations():
    """
    Endpoint unificado para obtener recomendaciones de productos.
    """
    global init_success, recommender
    
    if not init_success:
        init_success = init_app()
    
    if not init_success:
        return jsonify({"error": "Recommender system not initialized"}), 500
    
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "Request body is required"}), 400
        
        n_recommendations = data.get('n_recommendations', 5)
        
        if 'user_id' in data and data['user_id'] is not None:
            user_id = data['user_id']
            logger.info(f"Getting recommendations for user {user_id}")
            recommendations = recommender.get_recommendations(
                user_id,
                n_recommendations=n_recommendations
            )
        elif 'product_history' in data and data['product_history'] is not None:
            product_history = data['product_history']
            logger.info(f"Getting recommendations from product history: {product_history}")
            recommendations = recommender.get_recommendations_from_history(
                product_history,
                n_recommendations=n_recommendations
            )
        else:
            logger.info("Getting popular recommendations")
            recommendations = recommender._get_popular_recommendations(
                n_recommendations=n_recommendations
            )

        return jsonify({"recommendations": recommendations})
    except Exception as e:
        logger.error(f"Error getting recommendations: {str(e)}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
