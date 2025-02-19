from fastapi import FastAPI
import uvicorn
import os
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI()

@app.get("/")
async def root():
    return {"message": "API is running"}

@app.get("/test")
async def test():
    return {
        "status": "ok",
        "port": os.environ.get("PORT", "not set"),
        "message": "Test endpoint working"
    }

def start_server():
    try:
        port = int(os.environ.get("PORT", 10000))
        logger.info(f"Starting server on port {port}")
        uvicorn.run(
            app,
            host="0.0.0.0",
            port=port,
            log_level="info"
        )
    except Exception as e:
        logger.error(f"Error starting server: {e}")
        raise

if __name__ == "__main__":
    start_server()
