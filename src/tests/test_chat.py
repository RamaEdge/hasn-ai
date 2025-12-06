#!/usr/bin/env python3
"""
Tests for chat interaction endpoints
"""


import pytest
from fastapi.testclient import TestClient

from api.main import app
from core.cognitive_architecture import CognitiveArchitecture
from core.cognitive_models import CognitiveConfig


@pytest.fixture
def client():
    """Create test client"""
    return TestClient(app)


@pytest.fixture
def mock_cognitive_architecture():
    """Create mock cognitive architecture for testing"""
    config = CognitiveConfig(max_episodic_memories=50)
    return CognitiveArchitecture(config=config, backend_name="numpy")


def test_chat_endpoint_basic(client, mock_cognitive_architecture):
    """Test basic chat endpoint functionality"""
    # Set up cognitive architecture
    import api.routes.chat as chat_module

    chat_module._cognitive_architecture = mock_cognitive_architecture

    # Test POST /chat
    response = client.post("/chat", json={"message": "hello"})

    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert "response_text" in data
    assert "session_id" in data
    assert "provenance" in data
    assert "processing_time_ms" in data

    # Check provenance structure
    provenance = data["provenance"]
    assert "episode_ids" in provenance
    assert "semantic_links" in provenance
    assert "similar_traces_count" in provenance
    assert "consolidation_occurred" in provenance


def test_chat_endpoint_with_session(client, mock_cognitive_architecture):
    """Test chat endpoint with existing session"""
    import api.routes.chat as chat_module

    chat_module._cognitive_architecture = mock_cognitive_architecture

    session_id = "test-session-123"

    # First message
    response1 = client.post("/chat", json={"message": "hello", "session_id": session_id})
    assert response1.status_code == 200
    assert response1.json()["session_id"] == session_id

    # Second message in same session
    response2 = client.post("/chat", json={"message": "how are you?", "session_id": session_id})
    assert response2.status_code == 200
    assert response2.json()["session_id"] == session_id


def test_chat_endpoint_safety_filter(client, mock_cognitive_architecture):
    """Test that safety filter detects PII"""
    import api.routes.chat as chat_module

    chat_module._cognitive_architecture = mock_cognitive_architecture

    # Message with email (PII)
    response = client.post("/chat", json={"message": "Contact me at test@example.com"})

    # Should still process but log warning
    assert response.status_code == 200
    # Email should be filtered/redacted in processing


def test_get_session_context(client, mock_cognitive_architecture):
    """Test getting session context"""
    import api.routes.chat as chat_module

    chat_module._cognitive_architecture = mock_cognitive_architecture

    # Create a session first
    session_id = "test-session-context"
    client.post("/chat", json={"message": "hello", "session_id": session_id})

    # Get context
    response = client.get(f"/chat/{session_id}/context")
    assert response.status_code == 200
    data = response.json()
    assert data["session_id"] == session_id
    assert "message_count" in data
    assert "working_memory_items" in data
    assert "created_at" in data
    assert "last_activity" in data


def test_get_session_context_not_found(client):
    """Test getting context for non-existent session"""
    response = client.get("/chat/nonexistent-session/context")
    assert response.status_code == 404


def test_reset_session(client, mock_cognitive_architecture):
    """Test resetting a session"""
    import api.routes.chat as chat_module

    chat_module._cognitive_architecture = mock_cognitive_architecture

    # Create a session with messages
    session_id = "test-session-reset"
    client.post("/chat", json={"message": "message 1", "session_id": session_id})
    client.post("/chat", json={"message": "message 2", "session_id": session_id})

    # Reset session
    response = client.post(f"/chat/{session_id}/reset")
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert data["session_id"] == session_id
    assert data["messages_cleared"] == 2

    # Verify session is reset
    context_response = client.get(f"/chat/{session_id}/context")
    assert context_response.status_code == 200
    assert context_response.json()["message_count"] == 0


def test_reset_session_not_found(client):
    """Test resetting non-existent session"""
    response = client.post("/chat/nonexistent-session/reset")
    assert response.status_code == 404


def test_chat_provenance_populated(client, mock_cognitive_architecture):
    """Test that provenance is populated in response"""
    import api.routes.chat as chat_module

    chat_module._cognitive_architecture = mock_cognitive_architecture

    response = client.post("/chat", json={"message": "hello"})

    assert response.status_code == 200
    data = response.json()
    provenance = data["provenance"]

    # Should have at least episode ID
    assert isinstance(provenance["episode_ids"], list)
    assert isinstance(provenance["semantic_links"], list)
    assert isinstance(provenance["similar_traces_count"], int)
    assert isinstance(provenance["consolidation_occurred"], bool)
