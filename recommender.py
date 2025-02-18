import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import logging
from typing import Dict, List, Tuple, Optional
import gc
from scipy.sparse import csr_matrix, save_npz, load_npz
from scipy.stats import percentileofscore
import h5py
from pathlib import Path
from collections import Counter, defaultdict
import json

class ImprovedRecommender:
    def __init__(self, chunk_size: int = 1000):
        self.chunk_size = chunk_size
        self.setup_logging()
        self.data_dir = Path('recommender_data')
        self.data_dir.mkdir(exist_ok=True)

        # Constantes
        self.PRICE_RANGE_FACTOR = 0.3
        self.CATEGORY_BOOST = 0.4
        self.GENDER_BOOST = 0.5
        self.MIN_RATING_WEIGHT = 3

    def setup_logging(self):
        self.logger = logging.getLogger('ImprovedRecommender')
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)

    def _initialize_dimensions(self, products_df: pd.DataFrame):
        """Initialize feature dimensions and mappings"""
        self.main_categories = sorted(products_df['main_category'].unique())
        self.sub_categories = sorted(products_df['sub_category'].unique())

        self.numeric_dim = 5  # discount_price, actual_price, ratings, no_of_ratings, discount_percentage
        self.main_cat_dim = len(self.main_categories)
        self.total_dim = self.numeric_dim + self.main_cat_dim

        self.main_cat_map = {cat: idx for idx, cat in enumerate(self.main_categories)}
        self.n_products = len(products_df)

        self.logger.info(f"Features: {self.numeric_dim} numeric, {self.main_cat_dim} categories, {self.total_dim} total")

    def process_data(self, products_df: pd.DataFrame, interactions_df: pd.DataFrame):
        """Process and prepare all data"""
        self.logger.info("Starting data processing...")

        # Initialize dimensions
        self._initialize_dimensions(products_df)

        # Process and save product data
        self._process_products(products_df)

        # Process and save interaction data
        self._process_interactions(interactions_df)

        # Generate and save metadata
        self._save_metadata(products_df, interactions_df)

        self.logger.info("Data processing completed")

    def _process_products(self, products_df: pd.DataFrame):
        """Process and save product data"""
        self.logger.info("Processing products...")

        # Seleccionar columnas de información
        columns_to_save = ['product_id', 'name', 'main_category', 'sub_category',
                          'discount_price', 'ratings', 'no_of_ratings']
        if 'image_url' in products_df.columns:
            columns_to_save.append('image_url')
        if 'description' in products_df.columns:
            columns_to_save.append('description')

        products_df[columns_to_save].to_parquet(
            self.data_dir / 'products_info.parquet'
        )

        # Procesar características en chunks
        with h5py.File(self.data_dir / 'product_features.h5', 'w') as f:
            f.create_dataset('features', shape=(self.n_products, self.total_dim), dtype='float32')

            for i in range(0, len(products_df), self.chunk_size):
                chunk = products_df.iloc[i:i + self.chunk_size]
                features = self._create_product_features(chunk)
                f['features'][i:i + len(chunk)] = features

                if (i + self.chunk_size) % 10000 == 0:
                    self.logger.info(f"Processed {i + self.chunk_size} of {len(products_df)} products")
                    gc.collect()

    def _create_product_features(self, chunk: pd.DataFrame) -> np.ndarray:
        """Create feature vectors for a chunk of products"""
        features = np.zeros((len(chunk), self.total_dim))

        # Numeric features
        numeric_cols = ['discount_price', 'actual_price', 'ratings',
                       'no_of_ratings', 'discount_percentage']

        for i, col in enumerate(numeric_cols):
            values = chunk[col].fillna(0).values
            scaler = MinMaxScaler()
            features[:, i] = scaler.fit_transform(values.reshape(-1, 1)).flatten()

        # Categorical features
        for idx, row in chunk.iterrows():
            if row['main_category'] in self.main_cat_map:
                cat_idx = self.main_cat_map[row['main_category']] + self.numeric_dim
                features[idx - chunk.index[0], cat_idx] = 1

        return features

    def _process_interactions(self, interactions_df: pd.DataFrame):
        """Process and save user interactions"""
        self.logger.info("Processing user interactions...")

        # Crear mapeo de usuarios
        user_ids = sorted(interactions_df['user_id'].unique())
        self.user_id_map = {uid: idx for idx, uid in enumerate(user_ids)}
        np.save(self.data_dir / 'user_ids.npy', user_ids)

        # Procesar en chunks
        for i in range(0, len(user_ids), self.chunk_size):
            chunk_users = user_ids[i:i + self.chunk_size]
            chunk_data = interactions_df[interactions_df['user_id'].isin(chunk_users)]

            if len(chunk_data) > 0:
                matrix = self._create_interaction_matrix(chunk_data, chunk_users, i)
                save_npz(self.data_dir / f'interactions_{i}.npz', matrix)

            if (i + self.chunk_size) % 10000 == 0:
                self.logger.info(f"Processed {i + self.chunk_size} users")
                gc.collect()

        self.logger.info(f"Processed interactions for {len(user_ids)} users")

    def _create_interaction_matrix(self, chunk_data: pd.DataFrame,
                                 chunk_users: List[int], start_idx: int) -> csr_matrix:
        """Create interaction matrix for a chunk of users"""
        rows = []
        cols = []
        data = []

        for uid in chunk_users:
            user_interactions = chunk_data[chunk_data['user_id'] == uid]
            for _, row in user_interactions.iterrows():
                rows.append(self.user_id_map[uid] - start_idx)
                cols.append(row['product_id'])
                data.append(row['rating'])

        return csr_matrix(
            (data, (rows, cols)),
            shape=(len(chunk_users), self.n_products)
        )

    def _save_metadata(self, products_df: pd.DataFrame, interactions_df: pd.DataFrame):
        """Save metadata for recommendations"""
        self.logger.info("Saving metadata...")

        metadata = {
            'price_ranges': {
                'min': float(products_df['discount_price'].min()),
                'max': float(products_df['discount_price'].max()),
                'mean': float(products_df['discount_price'].mean()),
                'quartiles': products_df['discount_price'].quantile([0.25, 0.5, 0.75]).to_dict()
            },
            'gender_categories': self._extract_gender_categories(products_df),
            'category_stats': self._calculate_category_stats(products_df, interactions_df)
        }

        with open(self.data_dir / 'metadata.json', 'w') as f:
            json.dump(metadata, f)

    def _extract_gender_categories(self, products_df: pd.DataFrame) -> Dict[str, str]:
        """Extract gender information from categories"""
        gender_mapping = {}
        for category in products_df['main_category'].unique():
            category_lower = category.lower()
            if 'men' in category_lower and 'women' not in category_lower:
                gender_mapping[category] = 'male'
            elif 'women' in category_lower:
                gender_mapping[category] = 'female'
            else:
                gender_mapping[category] = 'neutral'
        return gender_mapping

    def _calculate_category_stats(self, products_df: pd.DataFrame,
                                interactions_df: pd.DataFrame) -> Dict:
        """Calculate category statistics"""
        stats = {}
        merged = interactions_df.merge(
            products_df[['product_id', 'main_category', 'discount_price']],
            on='product_id'
        )

        for category in products_df['main_category'].unique():
            cat_data = merged[merged['main_category'] == category]
            if len(cat_data) > 0:
                stats[category] = {
                    'avg_rating': float(cat_data['rating'].mean()),
                    'interaction_count': int(len(cat_data)),
                    'avg_price': float(cat_data['discount_price'].mean())
                }

        return stats

    def _get_user_preferences(self, user_id: int) -> Optional[Dict]:
        """Get user preferences based on interaction history with improved diversity"""
        try:
            chunk_idx = (self.user_id_map[user_id] // self.chunk_size) * self.chunk_size
            interactions = load_npz(self.data_dir / f'interactions_{chunk_idx}.npz')
            user_idx = self.user_id_map[user_id] - chunk_idx
            user_vector = interactions[user_idx].toarray().flatten()

            if user_vector.sum() == 0:
                return None

            products_info = pd.read_parquet(self.data_dir / 'products_info.parquet')
            products_info = products_info.dropna(subset=['discount_price', 'ratings'])

            rated_products = products_info[products_info.index.isin(np.where(user_vector > 0)[0])]

            if rated_products.empty:
                return None

            category_scores = defaultdict(float)
            interaction_counts = Counter(rated_products['main_category'])
            total_interactions = sum(interaction_counts.values())

            for cat, count in interaction_counts.items():
                cat_products = rated_products[rated_products['main_category'] == cat]
                avg_rating = cat_products['ratings'].mean()

                frequency_score = count / total_interactions
                rating_score = avg_rating / 5.0
                diversity_penalty = 1.0 - (frequency_score * 0.5)

                category_scores[cat] = (
                    frequency_score * 0.4 +
                    rating_score * 0.6
                ) * diversity_penalty

            related_categories = defaultdict(float)
            for cat, score in category_scores.items():
                cat_lower = cat.lower()

                for other_cat in products_info['main_category'].unique():
                    other_lower = other_cat.lower()

                    if other_cat == cat:
                        continue

                    related_keywords = {
                        'electronics': ['gadgets', 'accessories', 'devices'],
                        'clothing': ['wear', 'apparel', 'fashion'],
                        'shoes': ['footwear', 'sandals', 'boots'],
                        'accessories': ['fashion', 'style', 'wear']
                    }

                    for key, terms in related_keywords.items():
                        if key in cat_lower and any(term in other_lower for term in terms):
                            related_categories[other_cat] = max(
                                related_categories[other_cat],
                                score * 0.4
                            )

            final_categories = dict(category_scores)
            for cat, score in related_categories.items():
                if cat not in final_categories:
                    final_categories[cat] = score

            with open(self.data_dir / 'metadata.json', 'r') as f:
                metadata = json.load(f)

            return {
                'category_preferences': final_categories,
                'price_preferences': self._get_price_preferences(rated_products),
                'gender_preference': self._get_gender_preference(rated_products, metadata['gender_categories'])
            }

        except Exception as e:
            self.logger.error(f"Error loading user preferences: {str(e)}")
            return None

    def _get_category_preferences(self, rated_products: pd.DataFrame, ratings: np.ndarray) -> Dict[str, float]:
        """Calculate category preferences based on ratings"""
        category_scores = defaultdict(float)

        for cat in rated_products['main_category'].unique():
            cat_products = rated_products[rated_products['main_category'] == cat]
            cat_ratings = ratings[cat_products.index]

            avg_rating = np.mean(cat_ratings)
            count_weight = len(cat_ratings) / len(ratings)
            category_scores[cat] = avg_rating * count_weight

        if category_scores:
            max_score = max(category_scores.values())
            return {cat: score/max_score for cat, score in category_scores.items()}
        return {}

    def _get_price_preferences(self, rated_products: pd.DataFrame) -> Dict[str, float]:
        """Calculate price range preferences"""
        prices = rated_products['discount_price'].values
        return {
            'min': float(np.percentile(prices, 10)),
            'max': float(np.percentile(prices, 90)),
            'mean': float(np.mean(prices))
        }

    def _get_gender_preference(self, rated_products: pd.DataFrame,
                             gender_categories: Dict[str, str]) -> str:
        """Determine gender preference from purchase history"""
        gender_counts = Counter(
            gender_categories.get(cat, 'neutral')
            for cat in rated_products['main_category']
        )
        return max(gender_counts.items(), key=lambda x: x[1])[0]

    def _get_candidate_products(self, user_prefs: Dict) -> List[Dict]:
        """Get candidate products based on user preferences with improved diversity"""
        try:
            products_info = pd.read_parquet(self.data_dir / 'products_info.parquet')
            products_info = products_info.dropna(subset=['discount_price', 'ratings', 'no_of_ratings'])
            candidates = products_info.copy()

            # Score de precio (0-1)
            price_range = user_prefs['price_preferences']
            price_range_width = max(price_range['max'] - price_range['min'], 1)
            candidates['price_score'] = candidates['discount_price'].apply(
                lambda x: max(0, 1 - abs(x - price_range['mean']) / price_range_width)
            )

            # Score de categoría (0-1) con diversidad
            category_prefs = user_prefs['category_preferences']
            candidates['category_score'] = candidates['main_category'].apply(
                lambda x: category_prefs.get(x, 0.2)  # Score base 0.2 si no es preferida
            )

            if candidates['category_score'].max() > 0:
                candidates['category_score'] = candidates['category_score'] / candidates['category_score'].max()

            # Score de rating (0-1) con consideración de número de reviews
            review_weight = np.log1p(candidates['no_of_ratings']) / np.log1p(candidates['no_of_ratings'].max())
            candidates['rating_score'] = (
                (candidates['ratings'] / 5.0) * 0.7 +
                review_weight * 0.3
            )

            # Score de relevancia inicial
            candidates['relevance_score'] = (
                candidates['price_score'] * 0.3 +
                candidates['category_score'] * 0.4 +
                candidates['rating_score'] * 0.3
            )

            # Selección con diversidad
            recommendations = []
            used_categories = set()
            used_brands = set()

            # Pool inicial: top 100 según relevancia
            top_candidates = candidates.nlargest(100, 'relevance_score').copy()

            while len(recommendations) < 20 and not top_candidates.empty:
                top_candidates['final_score'] = top_candidates.apply(
                    lambda row: row['relevance_score'] * (
                        0.5 if row['main_category'] in used_categories else 1.0
                    ) * (
                        0.5 if row['name'].split()[0].lower() in used_brands else 1.0
                    ),
                    axis=1
                )

                # Boost aleatorio para diversidad
                random_boost = np.random.rand(len(top_candidates)) * 0.1
                top_candidates['final_score'] += random_boost

                best_idx = top_candidates['final_score'].idxmax()
                selected = top_candidates.loc[best_idx]

                recommendations.append({
                    'product_id': int(selected.name),
                    'name': selected['name'],
                    'category': selected['main_category'],
                    'price': float(selected['discount_price']),
                    'rating': float(selected['ratings']),
                    'score': float(selected['final_score']),
                    'image_url': selected.get('image_url', 'Imagen no disponible'),
                    'description': selected.get('description', 'Sin descripción')
                })

                used_categories.add(selected['main_category'])
                used_brands.add(selected['name'].split()[0].lower())

                same_category = top_candidates['main_category'] == selected['main_category']
                same_brand = top_candidates['name'].str.split().str[0].str.lower() == selected['name'].split()[0].lower()

                if len(recommendations) < 10:
                    top_candidates = top_candidates[~(same_category & same_brand)]
                else:
                    top_candidates = top_candidates[~(same_category | same_brand)]

            return recommendations

        except Exception as e:
            self.logger.error(f"Error getting candidate products: {str(e)}")
            return []

    def _calculate_product_score(self, product: pd.Series,
                               user_prefs: Dict,
                               product_features: np.ndarray,
                               selected_products: List[Dict] = None) -> float:
        """Calculate recommendation score for a product based on user preferences"""
        score = 0.0
        selected_products = selected_products or []

        # Puntuación por preferencia de categoría
        category = product['main_category']
        if category in user_prefs['category_preferences']:
            score += user_prefs['category_preferences'][category] * 2.0

        # Puntuación por preferencia de precio
        price = product['discount_price']
        price_prefs = user_prefs['price_preferences']
        price_delta = abs(price - price_prefs['mean'])
        price_range = price_prefs['max'] - price_prefs['min']
        if price_range > 0:
            price_score = 1.5 * (1.0 - min(price_delta / price_range, 1.0))
            score += price_score

        # Puntuación de rating con ponderación exponencial
        rating_weight = min(np.log1p(product['no_of_ratings']) / np.log1p(self.MIN_RATING_WEIGHT), 1.0)
        rating_score = 3.0 * (product['ratings'] / 5.0) * rating_weight
        score += rating_score

        # Puntuación por preferencia de género
        with open(self.data_dir / 'metadata.json', 'r') as f:
            metadata = json.load(f)
        product_gender = metadata['gender_categories'].get(category, 'neutral')
        if product_gender == user_prefs['gender_preference']:
            score += 1.0
        elif product_gender == 'neutral':
            score += 0.5

        # Penalización por diversidad
        for selected in selected_products:
            if selected['category'] == category:
                score *= 0.8
            product_brand = product['name'].split()[0].lower()
            selected_brand = selected['name'].split()[0].lower()
            if product_brand == selected_brand:
                score *= 0.7
            if abs(selected['price'] - price) < price_range * 0.1:
                score *= 0.9

        return score

    def _get_popular_recommendations(self, n_recommendations: int) -> List[Dict]:
        """Get diverse popular product recommendations when personalized recommendations are not available"""
        try:
            products_info = pd.read_parquet(self.data_dir / 'products_info.parquet')

            recommendations = []
            used_categories = set()
            used_brands = set()

            products_info['weighted_rating'] = (
                products_info['ratings'] *
                np.log1p(products_info['no_of_ratings'])
            )

            candidate_pool = products_info.copy()

            while len(recommendations) < n_recommendations and not candidate_pool.empty:
                candidate_pool['adjusted_score'] = candidate_pool.apply(
                    lambda row: row['weighted_rating'] * (
                        0.5 if row['main_category'] in used_categories else 1.0
                    ) * (
                        0.5 if row['name'].split()[0].lower() in used_brands else 1.0
                    ),
                    axis=1
                )

                best_idx = candidate_pool['adjusted_score'].idxmax()
                selected = candidate_pool.loc[best_idx]

                recommendations.append({
                    'product_id': int(best_idx),
                    'name': selected['name'],
                    'category': selected['main_category'],
                    'price': float(selected['discount_price']),
                    'rating': float(selected['ratings']),
                    'score': float(selected['weighted_rating']),
                    'image_url': selected.get('image_url', 'Imagen no disponible'),
                    'description': selected.get('description', 'Sin descripción')
                })

                used_categories.add(selected['main_category'])
                used_brands.add(selected['name'].split()[0].lower())

                same_brand_category = (
                    (candidate_pool['main_category'] == selected['main_category']) |
                    (candidate_pool['name'].str.split().str[0].str.lower() ==
                     selected['name'].split()[0].lower())
                )
                candidate_pool = candidate_pool[~same_brand_category]

            return recommendations

        except Exception as e:
            self.logger.error(f"Error getting popular recommendations: {str(e)}")
            return []

    def get_recommendations(self, user_id: int, n_recommendations: int = 5) -> List[Dict]:
        """Get personalized recommendations for a user"""
        try:
            user_prefs = self._get_user_preferences(user_id)
            if not user_prefs:
                self.logger.info(f"No preferences found for user {user_id}, using popular recommendations")
                return self._get_popular_recommendations(n_recommendations)

            candidates = self._get_candidate_products(user_prefs)
            if not candidates:
                self.logger.info(f"No candidates found for user {user_id}, using popular recommendations")
                return self._get_popular_recommendations(n_recommendations)

            return candidates[:n_recommendations]

        except Exception as e:
            self.logger.error(f"Error getting recommendations for user {user_id}: {str(e)}")
            return self._get_popular_recommendations(n_recommendations)

    def get_recommendations_from_history(self, product_ids: List[int], n_recommendations: int = 5) -> List[Dict]:
        """Get recommendations based on a manually provided purchase history"""
        try:
            products_info = pd.read_parquet(self.data_dir / 'products_info.parquet')
            rated_products = products_info[products_info['product_id'].isin(product_ids)]
            if rated_products.empty:
                self.logger.info("Historial de compras vacío. Se usarán recomendaciones populares.")
                return self._get_popular_recommendations(n_recommendations)

            # Calcular preferencias a partir del historial manual
            category_scores = defaultdict(float)
            interaction_counts = Counter(rated_products['main_category'])
            total_interactions = sum(interaction_counts.values())

            for cat, count in interaction_counts.items():
                cat_products = rated_products[rated_products['main_category'] == cat]
                avg_rating = cat_products['ratings'].mean()
                frequency_score = count / total_interactions
                rating_score = avg_rating / 5.0
                diversity_penalty = 1.0 - (frequency_score * 0.5)
                category_scores[cat] = (frequency_score * 0.4 + rating_score * 0.6) * diversity_penalty

            price_preferences = self._get_price_preferences(rated_products)
            with open(self.data_dir / 'metadata.json', 'r') as f:
                metadata = json.load(f)
            gender_preference = self._get_gender_preference(rated_products, metadata['gender_categories'])

            user_prefs = {
                'category_preferences': category_scores,
                'price_preferences': price_preferences,
                'gender_preference': gender_preference
            }

            candidates = self._get_candidate_products(user_prefs)
            if not candidates:
                return self._get_popular_recommendations(n_recommendations)
            return candidates[:n_recommendations]
        except Exception as e:
            self.logger.error(f"Error getting recommendations from history: {str(e)}")
            return self._get_popular_recommendations(n_recommendations)

    def run_interactive(self):
        """Run interactive recommendation interface"""
        while True:
            print("\nSeleccione una opción:")
            print("1. Ingresar ID de usuario")
            print("2. Ingresar historial de compras manualmente (IDs de productos separados por coma)")
            print("3. Salir")
            option = input("Opción: ").strip()

            if option == "1":
                try:
                    user_id = int(input("Ingrese el ID de usuario: ").strip())
                    recs = self.get_recommendations(user_id)
                except ValueError:
                    print("ID inválido. Intente nuevamente.")
                    continue
            elif option == "2":
                history = input("Ingrese los IDs de productos (separados por coma): ").strip()
                product_ids = [int(x.strip()) for x in history.split(",") if x.strip().isdigit()]
                recs = self.get_recommendations_from_history(product_ids)
            elif option == "3":
                print("Saliendo de la aplicación.")
                break
            else:
                print("Opción inválida, intente de nuevo.")
                continue

            print("\n=== Recomendaciones ===")
            for rec in recs:
                print(f"\nProducto: {rec['name']}")
                print(f"Descripción: {rec.get('description', 'Sin descripción')}")
                print(f"Imagen: {rec.get('image_url', 'Imagen no disponible')}")
                print(f"Categoría: {rec['category']}")
                print(f"Precio: ₹{rec['price']:.2f}")
                print(f"Rating: {rec['rating']:.1f}")
                print(f"Score: {rec['score']:.2f}")

    def cleanup(self):
        """Clean up temporary files"""
        if self.data_dir.exists():
            for file in self.data_dir.glob('*'):
                file.unlink()
            self.data_dir.rmdir()
