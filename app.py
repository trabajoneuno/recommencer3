from flask import Flask, request, jsonify
import pickle
import pandas as pd
from pathlib import Path
import os
import logging
import gc
import numpy as np
import sys
from recommender import ImprovedRecommender

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Global variables for model and data
recommender = None
products_df = None
interactions_df = None

def log_memory_usage(tag=""):
    """Log current memory usage using tracemalloc"""
    try:
        import resource
        memory_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        memory_mb = memory_kb / 1024  # Convert to MB
        logger.info(f"Memory usage {tag}: {memory_mb:.2f} MB")
    except ImportError:
        # If resource module is not available
        logger.info(f"Memory logging not available for: {tag}")

class CustomUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        # If pickle is looking for __main__.ImprovedRecommender, redirect to recommender.ImprovedRecommender
        if module == "__main__" and name == "ImprovedRecommender":
            module = "recommender"
        return super().find_class(module, name)

def init_app():
    """Initialize the application and load models with memory optimization"""
    global recommender, products_df, interactions_df
    
    try:
        logger.info("Starting model loading...")
        log_memory_usage("before loading")
        models_dir = Path("models")
        
        # Load DataFrames with optimization
        logger.info("Loading DataFrames...")
        
        # Optimize DataFrame loading - use appropriate dtypes
        try:
            with open(models_dir / 'products_df_dtypes.json', 'r') as f:
                import json
                dtypes = json.load(f)
                products_df = pd.read_pickle(models_dir / 'products_df.pkl')
                # Convert columns to optimized dtypes
                for col, dtype in dtypes.items():
                    if dtype.startswith('int'):
                        products_df[col] = pd.to_numeric(products_df[col], downcast='integer')
                    elif dtype.startswith('float'):
                        products_df[col] = pd.to_numeric(products_df[col], downcast='float')
                    elif dtype == 'category':
                        products_df[col] = products_df[col].astype('category')
        except (FileNotFoundError, json.JSONDecodeError):
            # Fallback if dtypes file doesn't exist
            products_df = pd.read_pickle(models_dir / 'products_df.pkl')
            # Automatically optimize common columns
            for col in products_df.columns:
                if products_df[col].dtype == 'object' and products_df[col].nunique() / len(products_df) < 0.5:
                    products_df[col] = products_df[col].astype('category')
        
        log_memory_usage("after loading products_df")
        
        # Similar optimization for interactions_df
        interactions_df = pd.read_pickle(models_dir / 'interactions_df.pkl')
        for col in interactions_df.select_dtypes(include=['int']).columns:
            interactions_df[col] = pd.to_numeric(interactions_df[col], downcast='integer')
        for col in interactions_df.select_dtypes(include=['float']).columns:
            interactions_df[col] = pd.to_numeric(interactions_df[col], downcast='float')
        
        log_memory_usage("after loading interactions_df")
        
        # Run garbage collection to free memory
        gc.collect()
        
        # Load recommender with a memory-efficient approach
        logger.info("Loading recommender...")
        with open(models_dir / 'recommender.pkl', 'rb') as f:
            recommender = CustomUnpickler(f).load()
        
        log_memory_usage("after loading recommender")
        
        # Initialize data directory
        logger.info("Initializing data directory...")
        recommender.data_dir = Path('recommender_data')
        recommender.data_dir.mkdir(exist_ok=True)
        
        # Update recommender with DataFrames (if needed by your implementation)
        recommender.products_df = products_df
        recommender.interactions_df = interactions_df
        
        # Run garbage collection again
        gc.collect()
        log_memory_usage("after initialization")
        
        logger.info("Models loaded successfully")
        return True
    except Exception as e:
        logger.error(f"Error loading models: {str(e)}")
        return False

# Modified ImprovedRecommender methods for batch processing
def patch_recommender():
    """Monkey patch the recommender with optimized methods"""
    global recommender
    
    original_get_recommendations_from_history = recommender.get_recommendations_from_history
    
    def optimized_get_recommendations_from_history(self, product_ids, n_recommendations=5):
        """Batch process recommendations to optimize memory usage"""
        # Process in batches to avoid memory issues
        batch_size = 10  # Adjust based on your memory constraints
        unique_products = list(set(product_ids))  # Remove duplicates
        
        if len(unique_products) <= batch_size:
            # If small enough, process normally
            return original_get_recommendations_from_history(unique_products, n_recommendations)
        
        # For larger lists, process in batches and merge results
        all_results = []
        for i in range(0, len(unique_products), batch_size):
            batch = unique_products[i:i+batch_size]
            log_memory_usage(f"before batch {i//batch_size}")
            batch_results = original_get_recommendations_from_history(batch, n_recommendations)
            all_results.extend(batch_results)
            # Clear memory
            gc.collect()
            log_memory_usage(f"after batch {i//batch_size}")
        
        # Deduplicate and sort by relevance (assuming recommender returns items with scores)
        # This part depends on your recommender's output format - adjust as needed
        if all_results and isinstance(all_results[0], tuple) and len(all_results[0]) == 2:
            # If results are (product_id, score) tuples
            unique_results = {}
            for product_id, score in all_results:
                if product_id not in unique_results or score > unique_results[product_id]:
                    unique_results[product_id] = score
            sorted_results = sorted(unique_results.items(), key=lambda x: x[1], reverse=True)
            return sorted_results[:n_recommendations]
        else:
            # If results are just product IDs
            unique_results = list(dict.fromkeys(all_results))  # Preserve order while removing duplicates
            return unique_results[:n_recommendations]
    
    # Replace the method
    recommender.get_recommendations_from_history = optimized_get_recommendations_from_history.__get__(recommender)

# Initialize Flask application
app = Flask(__name__)

# Try to load models at startup
init_success = init_app()
if init_success:
    patch_recommender()

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
    """Endpoint to verify API is functioning"""
    global init_success, recommender, products_df, interactions_df
    
    if not init_success:
        init_success = init_app()
        if init_success:
            patch_recommender()
    
    # Include basic memory info in health check
    try:
        import resource
        memory_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        memory_mb = memory_kb / 1024  # Convert to MB
    except ImportError:
        memory_mb = -1  # Not available
    
    return jsonify({
        "status": "healthy" if init_success else "error",
        "recommender_loaded": recommender is not None,
        "products_loaded": products_df is not None,
        "interactions_loaded": interactions_df is not None,
        "memory_usage_mb": memory_mb
    }), 200 if init_success else 500

@app.route('/recommend/user/<int:user_id>', methods=['GET'])
def get_user_recommendations(user_id):
    """Get recommendations for a specific user"""
    global init_success, recommender
    
    if not init_success:
        init_success = init_app()
        if init_success:
            patch_recommender()
    
    if not init_success:
        return jsonify({"error": "Recommender system not initialized"}), 500
    
    try:
        log_memory_usage(f"before user recs {user_id}")
        n_recommendations = int(request.args.get('n', 5))
        logger.info(f"Getting recommendations for user {user_id}")
        recommendations = recommender.get_recommendations(user_id, n_recommendations)
        log_memory_usage(f"after user recs {user_id}")
        
        # Force garbage collection
        gc.collect()
        
        return jsonify({
            "user_id": user_id,
            "recommendations": recommendations
        })
    except Exception as e:
        logger.error(f"Error getting recommendations for user {user_id}: {str(e)}")
        return jsonify({"error": str(e)}), 400

@app.route('/recommend/history', methods=['POST'])
def get_history_recommendations():
    """Get recommendations based on product history"""
    global init_success, recommender
    
    if not init_success:
        init_success = init_app()
        if init_success:
            patch_recommender()
    
    if not init_success:
        return jsonify({"error": "Recommender system not initialized"}), 500
    
    try:
        data = request.get_json()
        if not data or 'product_ids' not in data:
            return jsonify({"error": "product_ids list is required"}), 400
        
        product_ids = data['product_ids']
        n_recommendations = data.get('n', 5)
        
        # Log memory before processing
        log_memory_usage(f"before history recs {product_ids[:5]}...")
        
        logger.info(f"Getting recommendations from product history: {product_ids}")
        recommendations = recommender.get_recommendations_from_history(product_ids, n_recommendations)
        
        # Log memory after processing
        log_memory_usage(f"after history recs {product_ids[:5]}...")
        
        # Force garbage collection
        gc.collect()
        
        return jsonify({
            "product_history": product_ids,
            "recommendations": recommendations
        })
    except Exception as e:
        logger.error(f"Error getting recommendations from history: {str(e)}")
        return jsonify({"error": str(e)}), 400

@app.route('/recommend/popular', methods=['GET'])
def get_popular_recommendations():
    """Get popular recommendations"""
    global init_success, recommender
    
    if not init_success:
        init_success = init_app()
        if init_success:
            patch_recommender()
    
    if not init_success:
        return jsonify({"error": "Recommender system not initialized"}), 500
    
    try:
        log_memory_usage("before popular recs")
        n_recommendations = int(request.args.get('n', 5))
        logger.info("Getting popular recommendations")
        recommendations = recommender._get_popular_recommendations(n_recommendations)
        log_memory_usage("after popular recs")
        
        # Force garbage collection
        gc.collect()
        
        return jsonify({
            "recommendations": recommendations
        })
    except Exception as e:
        logger.error(f"Error getting popular recommendations: {str(e)}")
        return jsonify({"error": str(e)}), 400

@app.route('/recommend', methods=['POST'])
def get_recommendations():
    """
    Unified endpoint for product recommendations.
    """
    global init_success, recommender
    
    if not init_success:
        init_success = init_app()
        if init_success:
            patch_recommender()
    
    if not init_success:
        return jsonify({"error": "Recommender system not initialized"}), 500
    
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "Request body is required"}), 400
        
        n_recommendations = data.get('n_recommendations', 5)
        
        # Log memory before processing
        log_memory_usage("before unified recs")
        
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
        
        # Log memory after processing
        log_memory_usage("after unified recs")
        
        # Force garbage collection
        gc.collect()
        
        return jsonify({"recommendations": recommendations})
    except Exception as e:
        logger.error(f"Error getting recommendations: {str(e)}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
