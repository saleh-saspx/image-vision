import threading
from collections import OrderedDict

from app.models.schemas import NFTResponse

DEFAULT_MAX_SIZE = 512


class LRUCache:
    def __init__(self, max_size: int = DEFAULT_MAX_SIZE):
        self._cache: OrderedDict[str, NFTResponse] = OrderedDict()
        self._max_size = max_size
        self._lock = threading.Lock()

    def get(self, key: str) -> NFTResponse | None:
        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
                return self._cache[key]
            return None

    def set(self, key: str, value: NFTResponse) -> None:
        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
                self._cache[key] = value
            else:
                if len(self._cache) >= self._max_size:
                    self._cache.popitem(last=False)
                self._cache[key] = value

    def clear(self) -> None:
        with self._lock:
            self._cache.clear()

    @property
    def size(self) -> int:
        with self._lock:
            return len(self._cache)


cache = LRUCache()
