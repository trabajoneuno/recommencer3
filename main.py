import os
import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
async def startup_event():
    port = os.environ.get("PORT", 10000)
    logger.info(f"Application startup on port {port}")

@app.get("/")
async def root():
    return {"status": "ok"}

# Mock recommendations for initial testing
@app.get("/recomendaciones")
async def get_recommendations(product_name: str, top_n: int = 5):
    return {
        "producto": product_name,
        "recomendaciones": [
            {"id": 1, "name": "Test Product 1", "similarity": 0.9},
            {"id": 2, "name": "Test Product 2", "similarity": 0.8}
        ]
    }
