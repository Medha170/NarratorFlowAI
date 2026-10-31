import hashlib
import redis
from typing import Optional
from schemas.narrative import AnalysisResponse

try:
    redis_client = redis.Redis(host='localhost', port=6379, decode_responses=True)
    redis_client.ping()
    REDIS_AVAILABLE = True
    print("Redis cache connected sucessfully!")
except Exception as e:
    REDIS_AVAILABLE = False
    redis_client = None
    print(f"Redis unavailable ({e}), running without cache fallback.")

def generate_text_hash(text: str) -> str:
    """Takes input text and converts it into a secure, fixed-size SHA-256 hash key."""
    normalized_text = text.strip().lower()
    return hashlib.sha256(normalized_text.encode('utf-8')).hexdigest()

def get_cached_analysis(text_hash: str) -> Optional[AnalysisResponse]:
    """Retrieves and deserializes a cached analysis response from Redis if it exists."""
    if not REDIS_AVAILABLE or not redis_client:
        return None
    try:
        cached_json = redis_client.get(text_hash)
        if cached_json:
            return AnalysisResponse.model_validate_json(cached_json)
    except Exception as e:
        print(f"Cache read error: {e}")
    return None

def cache_analysis(text_hash: str, response: AnalysisResponse, expire_seconds: int=3600):
    """Serializes the Pydantic response model to JSON and stores it in Redis with a TTL (Time-To-Live)."""
    if not REDIS_AVAILABLE or not redis_client:
        return
    try:
        json_data = response.model_dump_json()
        redis_client.setex(text_hash, expire_seconds, json_data)
    except Exception as e:
        print(f"Cache write error: {e}")