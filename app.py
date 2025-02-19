from flask import Flask, request, jsonify
import pickle
import pandas as pd
from pathlib import Path
import os
from recommender import ImprovedRecommender

app = Flask(__name__)

# Cargar el modelo y datos al inicio
def load_models():
    models_dir = Path("models")
    with open(models_dir / 'recommender.pkl', 'rb') as f:
        recommender = pickle.load(f)
    products_df = pd.read_pickle(models_dir / 'products_df.pkl')
    interactions_df = pd.read_pickle(models_dir / 'interactions_df.pkl')
    return recommender, products_df, interactions_df

try:
    recommender, products_df, interactions_df = load_models()
except Exception as e:
    print(f"Error loading models: {e}")

@app.route('/health', methods=['GET'])
def health_check():
    """Endpoint para verificar que la API está funcionando"""
    return jsonify({"status": "healthy"}), 200

@app.route('/recommend/user/<int:user_id>', methods=['GET'])
def get_user_recommendations(user_id):
    """Obtener recomendaciones para un usuario específico"""
    try:
        n_recommendations = int(request.args.get('n', 5))
        recommendations = recommender.get_recommendations(user_id, n_recommendations)
        return jsonify({
            "user_id": user_id,
            "recommendations": recommendations
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route('/recommend/history', methods=['POST'])
def get_history_recommendations():
    """Obtener recomendaciones basadas en historial de productos"""
    try:
        data = request.get_json()
        if not data or 'product_ids' not in data:
            return jsonify({"error": "product_ids list is required"}), 400
            
        product_ids = data['product_ids']
        n_recommendations = data.get('n', 5)
        
        recommendations = recommender.get_recommendations_from_history(product_ids, n_recommendations)
        return jsonify({
            "product_history": product_ids,
            "recommendations": recommendations
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route('/recommend/popular', methods=['GET'])
def get_popular_recommendations():
    """Obtener recomendaciones populares"""
    try:
        n_recommendations = int(request.args.get('n', 5))
        recommendations = recommender._get_popular_recommendations(n_recommendations)
        return jsonify({
            "recommendations": recommendations
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
