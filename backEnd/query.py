import asyncio
import os
import sys
import signal
import logging
from datetime import datetime
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import importlib
import base64
from contextlib import asynccontextmanager
import uvicorn
import subprocess

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('query_server.log')
    ]
)
logger = logging.getLogger("query_server")


class QueryRequest(BaseModel):
    query: str


class ServerManager:
    def __init__(self):
        self.search_engine = None
        self.restart_flag = False
        self.start_time = datetime.now()

    async def initialize_engine(self):
        """Initialize the search engine"""
        logger.info("Initializing search engine...")
        self._load_search_engine()
        self.start_time = datetime.now()
        logger.info("Search engine initialized successfully")

    async def cleanup_engine(self):
        """Clean up the search engine"""
        if self.search_engine:
            logger.info("Cleaning up search engine...")
            self.search_engine.__exit__(None, None, None)
            self.search_engine = None
            logger.info("Search engine cleanup complete")

    def _load_search_engine(self):
        """Reload the module and reinitialize the search engine"""
        global OptimizedImageSearchEngine
        try:
            logger.info("Reloading OptimizedImageSearchEngine module...")
            importlib.reload(importlib.import_module("Sambanova_Langchain"))
            from Sambanova_Langchain import OptimizedImageSearchEngine
            self.search_engine = OptimizedImageSearchEngine()
            logger.info("OptimizedImageSearchEngine reloaded successfully!")
        except Exception as e:
            logger.error(f"Failed to reload OptimizedImageSearchEngine: {e}")
            self.search_engine = None

    async def reload_engine_periodically(self):
        """Reload the engine every 40 seconds"""
        while True:
            await asyncio.sleep(40)
            logger.info("Periodic reload: reinitializing the search engine...")
            await self.cleanup_engine()
            self._load_search_engine()

    async def check_health(self):
        """Check if the server is healthy"""
        return self.search_engine is not None


server_manager = ServerManager()


def restart_server():
    """Restart the server process"""
    logger.info("Restarting server...")
    try:
        cmd = [sys.executable] + sys.argv
        subprocess.Popen(cmd)
        os._exit(0)
    except Exception as e:
        logger.error(f"Failed to restart server: {e}")


async def health_check_loop():
    """Periodically check server health and restart if needed"""
    while True:
        await asyncio.sleep(5)
        if not await server_manager.check_health():
            restart_server()


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Setup signal handlers
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, lambda s, f: restart_server())

    # Initialize engine and start background tasks
    await server_manager.initialize_engine()
    asyncio.create_task(server_manager.reload_engine_periodically())
    asyncio.create_task(health_check_loop())

    yield

    # Cleanup
    await server_manager.cleanup_engine()


app = FastAPI(lifespan=lifespan)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    is_healthy = await server_manager.check_health()
    return {"status": "healthy" if is_healthy else "restarting"}


@app.post("/query")
async def process_query(request: QueryRequest):
    """Process incoming queries"""
    if server_manager.restart_flag:
        raise HTTPException(status_code=503, detail="Server is restarting")

    if not server_manager.search_engine:
        await server_manager.initialize_engine()

    query = request.query
    logger.info(f"Processing query: {query}")

    try:
        user_response, selected_images = await server_manager.search_engine.search_and_respond(query)
        images_with_data = []

        for img in selected_images:
            try:
                with open(img["path"], "rb") as img_file:
                    image_data = base64.b64encode(img_file.read()).decode("utf-8")
                images_with_data.append({
                    "path": img["path"],
                    "data": f"data:image/jpeg;base64,{image_data}"
                })
            except Exception as e:
                logger.error(f"Error processing image {img['path']}: {e}")

        return {
            "answer": user_response,
            "relevant_images": images_with_data
        }
    except Exception as e:
        logger.error(f"Query processing error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    try:
        uvicorn.run(
            "query:app",
            host="0.0.0.0",
            port=9000,
            reload=False,
            log_level="info"
        )
    except Exception as e:
        logger.error(f"Server error: {e}")
        sys.exit(1)