import os
import unittest
from datetime import datetime
from types import SimpleNamespace

import chat
from conversation_memory import RetrievedMemory, RetrievedUserMemory


class FakeResponses:
    def __init__(self):
        self.kwargs = None

    def create(self, **kwargs):
        self.kwargs = kwargs
        return SimpleNamespace(id="resp_test", output_text="Test answer")


class FakeOpenAI:
    def __init__(self):
        self.responses = FakeResponses()


class FakeConversationMemory:
    def __init__(self):
        self.remembered = []
        self.agent_id = "00000000-0000-0000-0000-000000000123"
        self.resolved_users = []
        self.user_memory_queries = []
        self.conversation_queries = []

    def get_or_create_default_agent(self, *, user_id):
        self.resolved_users.append(user_id)
        return self.agent_id

    def retrieve_user_memories(self, client, *, agent_id, user_id, query):
        self.user_memory_queries.append(
            {"agent_id": agent_id, "user_id": user_id, "query": query}
        )
        return [
            RetrievedUserMemory(
                type="preference",
                content="User prefers concise explanations.",
                confidence=0.9,
                similarity=0.8,
                created_at=datetime.now(),
            )
        ]

    def retrieve(self, client, *, agent_id, conversation_id, query):
        self.conversation_queries.append(
            {"agent_id": agent_id, "conversation_id": conversation_id, "query": query}
        )
        return [
            RetrievedMemory(
                role="user",
                content="The user recently asked about Supabase memory.",
                similarity=0.7,
                created_at=datetime.now(),
            )
        ]

    def remember(self, client, **kwargs):
        self.remembered.append(kwargs)


class MemoryContextTest(unittest.TestCase):
    def setUp(self):
        previous_memory = chat.conversation_memory
        previous_summary_enabled = os.environ.get("MEMORY_SUMMARY_ENABLED")
        previous_web_search_enabled = os.environ.get("WEB_SEARCH_ENABLED")

        os.environ["MEMORY_SUMMARY_ENABLED"] = "false"
        os.environ["WEB_SEARCH_ENABLED"] = "false"
        self.memory = FakeConversationMemory()
        chat.conversation_memory = self.memory
        self.addCleanup(
            self.restore_state,
            previous_memory,
            previous_summary_enabled,
            previous_web_search_enabled,
        )

    def restore_state(
        self,
        previous_memory,
        previous_summary_enabled,
        previous_web_search_enabled,
    ):
        chat.conversation_memory = previous_memory
        if previous_summary_enabled is None:
            os.environ.pop("MEMORY_SUMMARY_ENABLED", None)
        else:
            os.environ["MEMORY_SUMMARY_ENABLED"] = previous_summary_enabled
        if previous_web_search_enabled is None:
            os.environ.pop("WEB_SEARCH_ENABLED", None)
        else:
            os.environ["WEB_SEARCH_ENABLED"] = previous_web_search_enabled

    def test_complete_chat_turn_includes_saved_memories_and_recent_chats(self):
        client = FakeOpenAI()
        response = chat.complete_chat_turn(
            user_message="What should we do next?",
            conversation_id="default",
            previous_response_id=None,
            user=chat.AuthenticatedUser(id="user_123", email="user@example.com"),
            client=client,
        )

        self.assertEqual(response.message, "Test answer")
        instructions = client.responses.kwargs["instructions"]
        self.assertIn("Relevant durable user memories", instructions)
        self.assertIn("User prefers concise explanations.", instructions)
        self.assertIn("Relevant previous conversation details", instructions)
        self.assertIn("recently asked about Supabase memory", instructions)

    def test_complete_chat_turn_uses_agent_id_for_memory_scope(self):
        client = FakeOpenAI()
        chat.complete_chat_turn(
            user_message="Remember this for later.",
            conversation_id="project-alpha",
            previous_response_id=None,
            user=chat.AuthenticatedUser(id="user_123", email="user@example.com"),
            client=client,
        )

        self.assertEqual(self.memory.resolved_users, ["user_123"])
        self.assertEqual(
            self.memory.user_memory_queries[0]["agent_id"],
            self.memory.agent_id,
        )
        self.assertEqual(
            self.memory.conversation_queries[0]["agent_id"],
            self.memory.agent_id,
        )
        self.assertTrue(
            all(
                item["agent_id"] == self.memory.agent_id
                for item in self.memory.remembered
            )
        )

    def test_complete_chat_turn_stores_raw_conversation_id(self):
        client = FakeOpenAI()
        response = chat.complete_chat_turn(
            user_message="Use the raw conversation id.",
            conversation_id="local",
            previous_response_id=None,
            user=chat.AuthenticatedUser(id="user_123", email="user@example.com"),
            client=client,
        )

        self.assertEqual(response.conversation_id, "local")
        self.assertEqual(
            [item["conversation_id"] for item in self.memory.remembered],
            ["local", "local"],
        )
        self.assertNotIn(
            "user_123:local",
            [item["conversation_id"] for item in self.memory.remembered],
        )


if __name__ == "__main__":
    unittest.main()
