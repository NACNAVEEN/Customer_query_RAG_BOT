"""Tests for conversation memory and semantic cache."""

from app.cache.semantic_cache import SemanticCache
from app.memory.conversation_memory import ConversationBufferMemory


class TestConversationMemory:
    def test_max_interactions_limit(self) -> None:
        memory = ConversationBufferMemory(max_interactions=3)
        for i in range(5):
            memory.add(f"question {i}", f"answer {i}")
        assert len(memory) == 3
        history = memory.get_history()
        assert history[0]["user"] == "question 2"

    def test_clear_memory(self) -> None:
        memory = ConversationBufferMemory()
        memory.add("q", "a")
        memory.clear()
        assert len(memory) == 0


class TestSemanticCache:
    def test_cache_miss(self) -> None:
        cache = SemanticCache()
        cache.enabled = True
        cache._redis_client = None
        cache._memory_store.clear()
        result = cache.lookup("unique query that won't match")
        assert not result.hit

    def test_cache_store_and_lookup(self) -> None:
        cache = SemanticCache()
        cache.enabled = True
        cache._redis_client = None
        cache._memory_store.clear()
        cache.threshold = 0.90

        query = "what is instapark anpr technology"
        cache.store(query, "ANPR is Automatic Number Plate Recognition.", citations=[])

        result = cache.lookup(query)
        assert result.hit
        assert "ANPR" in result.response
