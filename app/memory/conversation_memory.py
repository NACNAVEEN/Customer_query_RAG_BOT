"""Conversation buffer memory for chat history."""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass, field
from typing import Any

from app.config.settings import get_settings

logger = logging.getLogger(__name__)


@dataclass
class ConversationTurn:
    """A single user-assistant exchange."""

    user: str
    assistant: str
    metadata: dict[str, Any] = field(default_factory=dict)


class ConversationBufferMemory:
    """Store last N conversation interactions."""

    def __init__(self, max_interactions: int | None = None) -> None:
        settings = get_settings()
        self.max_interactions = max_interactions or settings.memory_max_interactions
        self._turns: deque[ConversationTurn] = deque(maxlen=self.max_interactions)

    def add(self, user: str, assistant: str, metadata: dict[str, Any] | None = None) -> None:
        """Add a conversation turn."""
        self._turns.append(
            ConversationTurn(
                user=user,
                assistant=assistant,
                metadata=metadata or {},
            )
        )
        logger.debug("Memory updated, storing %d turns", len(self._turns))

    def get_history(self) -> list[dict[str, str]]:
        """Return chat history for LLM context."""
        return [{"user": t.user, "assistant": t.assistant} for t in self._turns]

    def get_turns(self) -> list[ConversationTurn]:
        return list(self._turns)

    def clear(self) -> None:
        self._turns.clear()
        logger.info("Conversation memory cleared")

    def __len__(self) -> int:
        return len(self._turns)
