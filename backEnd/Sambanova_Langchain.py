# Fix OpenMP error
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# Imports
import base64
import easyocr
import faiss
import json
import numpy as np
import torch
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from dotenv import load_dotenv
from functools import lru_cache, wraps
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from PIL import Image
import PIL
from tqdm import tqdm
from transformers import CLIPModel, CLIPProcessor
from typing import List, Dict, Any, Tuple
import asyncio
import platform
import sys

# Prevent DecompressionBomb errors
PIL.Image.MAX_IMAGE_PIXELS = None

# Load environment variables
load_dotenv()
SAMBANOVA_API_KEY = os.getenv("SAMBANOVA_API_KEY")
if not SAMBANOVA_API_KEY:
    raise ValueError("Please set SAMBANOVA_API_KEY in .env file")

# Configuration
IMAGE_DIRECTORY = "faiss/"
FAISS_INDEX_FILE = "faiss_index.index"
METADATA_FILE = "metadata.json"
EASYOCR_MODEL_DIR = "easyocr_models/"
# Ensure the image directory exists
os.makedirs(IMAGE_DIRECTORY, exist_ok=True)

# Set device and optimize torch
device = torch.device('mps' if torch.backends.mps.is_available() else
                      ('cuda' if torch.cuda.is_available() else 'cpu'))
torch.set_num_threads(4)

# Configure logging
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    stream=sys.stdout
)
logger = logging.getLogger(__name__)


def log_execution(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        print(f"Executing {func.__name__}...")
        result = func(*args, **kwargs)
        print(f"Completed {func.__name__}")
        return result

    return wrapper


class OptimizedImageSearchEngine:
    def __init__(self):
        print(f"Initializing search engine using device: {device}")
        self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
        self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self.reader = easyocr.Reader(['en'], gpu=torch.cuda.is_available(), download_enabled=False, model_storage_directory="EASYOCR_MODEL_DIR")
        self.dimension = 512
        self.llm = ChatOpenAI(
            temperature=0.7,
            model="Llama-3.2-90B-Vision-Instruct",
            api_key=SAMBANOVA_API_KEY,
            base_url="https://api.sambanova.ai/v1"
        )
        # Initialize executor only when needed
        self.executor = None
        self.load_or_create_index()
        self.setup_chains()

    def __enter__(self):
        self.executor = ThreadPoolExecutor(max_workers=4)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.executor:
            self.executor.shutdown(wait=False)
            self.executor = None

    def setup_chains(self):
        """Setup LangChain chains using the latest methods"""
        # Analysis prompt
        self.analysis_prompt = PromptTemplate.from_template(
            """You are an AI-powered Image Search Engine Assistant. Analyze these images and their extracted text to find matches for the user's query.

            Query: {query}

            Top 6 candidate images:
            {results}

            Analyze each image and its extracted text carefully. Select ONLY the images that have ANY relevant connection to the query, even if it's just mentioning the name or concept.

            Return ONLY the filenames of relevant images, separated by commas. Strictly Do not write the images filepath names of images which are not qualified and relevant. If no images are relevant, return "No relevant images found."

            Important: Be confident in your choices and explain WHY each selected image is relevant to the query."""
        )

        # Response prompt
        self.response_prompt = PromptTemplate.from_template(
            """Given the query: "{query}"
            And these images: {images}

            Generate ONE CONCISE sentence that directly answers the question strictly in a single sentence.
            MUST be ONE natural-sounding sentence."""
        )

        # Setup chains using the new method
        self.analysis_chain = (
                self.analysis_prompt
                | self.llm
                | StrOutputParser()
        )

        self.response_chain = (
                self.response_prompt
                | self.llm
                | StrOutputParser()
        )

    def load_or_create_index(self):
        """Load existing FAISS index or create new one"""
        if os.path.exists(FAISS_INDEX_FILE):
            print("Loading existing FAISS index...")
            self.index = faiss.read_index(FAISS_INDEX_FILE)
            with open(METADATA_FILE, 'r') as f:
                self.metadata = json.load(f)
            print(f"Loaded index with {len(self.metadata)} images")
        else:
            print("Creating new FAISS index...")
            self.index = faiss.IndexFlatL2(self.dimension)
            self.metadata = []

    def save_index(self):
        """Save FAISS index and metadata"""
        faiss.write_index(self.index, FAISS_INDEX_FILE)
        with open(METADATA_FILE, 'w') as f:
            json.dump(self.metadata, f)
        print("Index and metadata saved successfully")

    @lru_cache(maxsize=128)
    def preprocess_image(self, image_path: str) -> Image.Image:
        """Cached image preprocessing"""
        with Image.open(image_path) as img:
            img = img.convert("RGB")
            return img.resize((224, 224), Image.Resampling.LANCZOS)

    @torch.no_grad()
    def get_image_embeddings(self, image_path: str) -> np.ndarray:
        """Optimized image embedding generation"""
        try:
            image = self.preprocess_image(image_path)
            inputs = self.clip_processor(images=image, return_tensors="pt").to(device)
            embeddings = self.clip_model.get_image_features(**inputs)
            return embeddings.cpu().numpy()
        except Exception as e:
            print(f"Error generating embeddings for {image_path}: {str(e)}")
            return np.zeros((1, self.dimension))

    @torch.no_grad()
    def extract_text_and_embeddings(self, image_path: str) -> Tuple[str, np.ndarray]:
        """Extract text and generate text embeddings"""
        try:
            # Extract text using EasyOCR
            raw_text = " ".join(self.reader.readtext(image_path, detail=0))

            # Process text through CLIP
            inputs = self.clip_processor(
                text=[raw_text if raw_text else "no text found"],
                padding=True,
                truncation=True,
                max_length=77,
                return_tensors="pt"
            ).to(device)

            text_embeddings = self.clip_model.get_text_features(**inputs)
            return raw_text, text_embeddings.cpu().numpy()
        except Exception as e:
            print(f"Error processing text for {image_path}: {str(e)}")
            return "", np.zeros((1, self.dimension))

    def process_single_image(self, img_path: str) -> Dict[str, Any]:
        """Process a single image with optimized memory usage"""
        try:
            image_emb = self.get_image_embeddings(img_path)
            text, text_emb = self.extract_text_and_embeddings(img_path)
            return {
                'path': img_path,
                'embeddings': (image_emb + text_emb) / 2,
                'text': text
            }
        except Exception as e:
            print(f"Error processing {img_path}: {str(e)}")
            return None

    def batch_process_images(self, image_paths: List[str], batch_size: int = 4) -> List[Dict[str, Any]]:
        """Process images in batches"""
        if not self.executor:
            self.executor = ThreadPoolExecutor(max_workers=4)

        try:
            results = []
            for i in range(0, len(image_paths), batch_size):
                batch = image_paths[i:i + batch_size]
                futures = [self.executor.submit(self.process_single_image, path)
                           for path in batch]
                results.extend([f.result() for f in futures if f.result() is not None])
            return results
        finally:
            # Clean up executor after batch processing
            self.executor.shutdown(wait=True)
            self.executor = None

    @log_execution
    def update_index(self) -> bool:
        """
        Process and update index one image at a time.
        Returns True if new images were processed, False otherwise.
        """
        existing_paths = {item['path'] for item in self.metadata}
        image_files = [
            os.path.join(IMAGE_DIRECTORY, f) for f in os.listdir(IMAGE_DIRECTORY)
            if f.lower().endswith(('.png', '.jpg', '.jpeg'))
        ]

        new_images = [img for img in image_files if img not in existing_paths]

        if not new_images:
            print("No new images found to process.")
            return False

        print(f"Found {len(new_images)} new images to process...")

        for i, image_path in enumerate(new_images, 1):
            print(f"Processing image {i} of {len(new_images)}: {os.path.basename(image_path)}")

            # Process single image
            processed_result = self.process_single_image(image_path)

            if processed_result:
                # Add single embedding to index
                embedding = processed_result['embeddings']
                self.index.add(embedding)

                # Add metadata for single image
                self.metadata.append({
                    'path': processed_result['path'],
                    'text': processed_result['text'],
                    'timestamp': datetime.fromtimestamp(
                        os.path.getmtime(processed_result['path'])
                    ).strftime('%Y-%m-%d %H:%M:%S')
                })

                # Save index and metadata after each image
                self.save_index()
                print(f"✓ Added image {i} to index and saved")
            else:
                print(f"✗ Failed to process image {i}: {os.path.basename(image_path)}")

        print(f"Completed processing all {len(new_images)} new images")
        return True

    def search_images(self, query: str, k: int = 6) -> List[Dict[str, Any]]:
        """Search for images based on query"""
        query_inputs = self.clip_processor(
            text=[query],
            padding=True,
            truncation=True,
            max_length=77,
            return_tensors="pt"
        ).to(device)

        with torch.no_grad():
            query_embeddings = self.clip_model.get_text_features(**query_inputs)
            query_embeddings = query_embeddings.cpu().numpy()

        distances, indices = self.index.search(query_embeddings, k)

        return [
            {
                "path": self.metadata[idx]["path"],
                "text": self.metadata[idx]["text"],
                "score": float(distances[0][i])
            }
            for i, idx in enumerate(indices[0])
        ]

    async def analyze_results_with_llm(self, query: str, results: List[Dict[str, Any]]) -> Tuple[
        List[Dict[str, Any]], str]:
        """Analyze search results using SambaNova LLM with new LangChain methods"""
        formatted_results = "\n".join([
            f"Image {i + 1}: {os.path.basename(r['path'])}\n"
            f"Text: {r['text']}\n"
            f"Score: {r['score']}"
            for i, r in enumerate(results)
        ])

        response = await self.analysis_chain.ainvoke({
            "query": query,
            "results": formatted_results
        })

        selected_images = [
            result for result in results
            if os.path.basename(result['path']) in response
        ]

        return selected_images, response

    async def generate_user_response(self, query: str, selected_images: List[Dict[str, Any]]) -> str:
        """Generate a user-friendly response using SambaNova LLM"""
        formatted_images = "\n".join([
            f"Image {img['text']}"
            for img in selected_images
        ])

        return await self.response_chain.ainvoke({
            "query": query,
            "images": formatted_images
        })

    async def search_and_respond(self, query: str) -> Tuple[str, List[Dict[str, Any]]]:
        """Combined search and response generation"""
        results = self.search_images(query)
        selected_images, _ = await self.analyze_results_with_llm(query, results)
        user_response = await self.generate_user_response(query, selected_images)
        return user_response, selected_images

    def display_results(self, user_response: str, selected_images: List[Dict[str, Any]]):
        """Enhanced result display"""
        print("\nAnswer:", user_response)
        if selected_images:
            print("\nRelevant Image(s):")
            for img in selected_images:
                print(f"Image: {os.path.basename(img['path'])}")
        else:
            print("\nNo relevant images found.")


async def update_index_periodically():
    while True:
        try:
            # Run the embedding and FAISS update every 3 minutes
            with OptimizedImageSearchEngine() as search_engine:
                search_engine.update_index()  # Update FAISS index with individual processing
            print("Index update cycle completed successfully.")
        except Exception as e:
            print(f"Error during index update: {str(e)}")

        # Sleep for 3 minutes before checking for new images again
        await asyncio.sleep(45)


async def main():
    """Main function that runs once"""
    try:
        with OptimizedImageSearchEngine() as search_engine:
            search_engine.update_index()
        print("Index update completed successfully.")
    except Exception as e:
        print(f"Error during index update: {str(e)}")

if __name__ == "__main__":
    # Fix for Windows AsyncIO
    if platform.system() == 'Windows':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

    try:
        asyncio.run(main())
        logger.info("Application completed successfully")
    except KeyboardInterrupt:
        logger.info("Application stopped by user")
    except Exception as e:
        logger.error(f"Application failed: {str(e)}")
        sys.exit(1)
    finally:
        # Clean exit
        sys.exit(0)