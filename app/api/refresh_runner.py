import asyncio
from index import load_cache, save_cache, refresh_cache

if __name__ == "__main__":
    load_cache()                       # load existing cache from disk
    asyncio.run(refresh_cache())       # refresh with new data
    save_cache()                       # persist cache to file