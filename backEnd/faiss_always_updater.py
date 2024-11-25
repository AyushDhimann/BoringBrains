# faiss_always_updater.py
import os
import asyncio
from datetime import datetime
from Sambanova_Langchain import OptimizedImageSearchEngine


async def continuous_monitoring():
    """Continuously monitor for new images and process them"""
    print("Starting continuous image monitoring...")
    while True:
        try:
            with OptimizedImageSearchEngine() as search_engine:
                # Check and process new images
                images_processed = search_engine.update_index()

                if images_processed:
                    print("Completed processing batch. Checking for new images immediately...")
                else:
                    print("No new images found. Checking again...")
                await asyncio.sleep(1)  # Small delay for responsiveness

        except Exception as e:
            print(f"Error during monitoring cycle: {str(e)}")
            await asyncio.sleep(5)  # Pause on error to prevent rapid loops


if __name__ == "__main__":
    try:
        asyncio.run(continuous_monitoring())
    except KeyboardInterrupt:
        print("Monitoring stopped by user.")
