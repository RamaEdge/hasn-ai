"""
Test vision interaction endpoints with toy embeddings
"""

import pytest
from fastapi.testclient import TestClient

from api.main import app


@pytest.fixture
def client():
    """Create test client"""
    return TestClient(app)


def test_vision_ingest_and_query():
    """Test ingesting and querying vision embeddings"""
    client = TestClient(app)

    # Create toy embeddings (simulating CLIP embeddings)
    # Embedding 1: "cat" concept
    cat_embedding = [0.1, 0.2, 0.3, 0.4, 0.5] * 20  # 100-dim embedding
    # Embedding 2: "dog" concept (similar to cat)
    dog_embedding = [0.15, 0.25, 0.35, 0.45, 0.55] * 20  # 100-dim embedding
    # Embedding 3: "car" concept (different from cat/dog)
    car_embedding = [0.9, 0.8, 0.7, 0.6, 0.5] * 20  # 100-dim embedding

    # Ingest embeddings
    response1 = client.post(
        "/vision/ingest",
        json={
            "embedding": cat_embedding,
            "context": {"label": "cat", "source": "test"},
        },
    )
    assert response1.status_code == 200
    data1 = response1.json()
    assert data1["success"] is True
    assert "trace_id" in data1
    trace_id_1 = data1["trace_id"]

    response2 = client.post(
        "/vision/ingest",
        json={
            "embedding": dog_embedding,
            "context": {"label": "dog", "source": "test"},
        },
    )
    assert response2.status_code == 200
    data2 = response2.json()
    assert data2["success"] is True
    trace_id_2 = data2["trace_id"]

    response3 = client.post(
        "/vision/ingest",
        json={
            "embedding": car_embedding,
            "context": {"label": "car", "source": "test"},
        },
    )
    assert response3.status_code == 200
    data3 = response3.json()
    assert data3["success"] is True
    trace_id_3 = data3["trace_id"]

    # Query with cat embedding - should find cat and dog (similar)
    query_response = client.post(
        "/vision/query",
        json={"embedding": cat_embedding, "context": {}},
        params={"limit": 10, "similarity_threshold": 0.5},
    )
    assert query_response.status_code == 200
    query_data = query_response.json()
    assert query_data["success"] is True
    assert query_data["count"] > 0

    # Check that we found at least the cat embedding
    trace_ids = [result["trace_id"] for result in query_data["results"]]
    assert trace_id_1 in trace_ids

    # Query with car embedding - should find car
    query_response2 = client.post(
        "/vision/query",
        json={"embedding": car_embedding, "context": {}},
        params={"limit": 10, "similarity_threshold": 0.5},
    )
    assert query_response2.status_code == 200
    query_data2 = query_response2.json()
    assert query_data2["success"] is True
    assert query_data2["count"] > 0

    # Check that we found the car embedding
    trace_ids2 = [result["trace_id"] for result in query_data2["results"]]
    assert trace_id_3 in trace_ids2


def test_vision_ingest_empty_embedding():
    """Test that empty embedding is rejected"""
    client = TestClient(app)

    response = client.post(
        "/vision/ingest",
        json={"embedding": [], "context": {}},
    )
    assert response.status_code == 400


def test_vision_query_empty_embedding():
    """Test that empty query embedding is rejected"""
    client = TestClient(app)

    response = client.post(
        "/vision/query",
        json={"embedding": [], "context": {}},
    )
    assert response.status_code == 400


if __name__ == "__main__":
    # Simple manual test
    print("Testing vision endpoints...")
    test_vision_ingest_and_query()
    print("✓ All tests passed!")

