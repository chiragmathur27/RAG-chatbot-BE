from langchain_community.chat_message_histories import ChatMessageHistory
from langchain.schema import BaseMessage, HumanMessage, AIMessage
import redis
import json
from datetime import datetime
from typing import List

class RedisChatMessageHistory(ChatMessageHistory):
    
    def __init__(self, session_id: str, url="redis://localhost:6379"):
        super().__init__()
        self._session_id = session_id
        self._redis = redis.Redis.from_url(url, decode_responses=True)
        self._key = f"chat:{session_id}"
        self._load_history()

    def _load_history(self):

        history = self._redis.get(self._key)
        if history:
            messages_data = json.loads(history)
            # Clear any existing messages first
            self.messages.clear()
            # Add messages from Redis
            for msg in messages_data:
                if msg["role"] == "user":
                    self.messages.append(HumanMessage(content=msg["content"]))
                else:
                    self.messages.append(AIMessage(content=msg["content"]))

    def add_message(self, message: BaseMessage) -> None:
        """Add a message to the history and persist to Redis."""
        super().add_message(message)
        self._save_to_redis()
        
    def add_user_message(self, message: str) -> None:
        """Add a user message to the history."""
        self.add_message(HumanMessage(content=message))
        
    def add_ai_message(self, message: str) -> None:
        """Add an AI message to the history."""
        self.add_message(AIMessage(content=message))
        
    def _save_to_redis(self) -> None:
        """Save messages to Redis."""
        messages_data = []
        for msg in self.messages:
            role = "user" if isinstance(msg, HumanMessage) else "assistant"
            messages_data.append({
                "role": role,
                "content": msg.content,
                "timestamp": datetime.now().isoformat()
            })
        
        self._redis.set(
            self._key,
            json.dumps(messages_data),
            ex=86400  # Expire after 24 hours
        )

    def clear(self) -> None:
        """Clear messages from memory and Redis."""
        super().clear()
        self._redis.delete(self._key)