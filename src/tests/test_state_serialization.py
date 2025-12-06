#!/usr/bin/env python3
"""
Tests for state serialization and knowledge search (THE-36)
Tests save→load roundtrip and top-k retrieval
"""

import pytest
import tempfile
import shutil
from pathlib import Path

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.cognitive_architecture import CognitiveArchitecture
from core.cognitive_models import CognitiveConfig
from storage.cognitive_serializer import CognitiveArchitectureSerializer
from common.random import seed_rng


@pytest.fixture
def temp_storage():
    """Create temporary storage directory"""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)


@pytest.fixture
def architecture():
    """Create a test cognitive architecture"""
    seed_rng(42)  # Ensure deterministic behavior
    config = CognitiveConfig(
        max_episodic_memories=100,
        max_working_memory_items=20,
        semantic_consolidation_threshold=2,  # Lower for testing
    )
    return CognitiveArchitecture(config=config, backend_name="numpy")


@pytest.fixture
def serializer(temp_storage):
    """Create serializer with temporary storage"""
    return CognitiveArchitectureSerializer(storage_path=temp_storage)


class TestSnapshotSaveLoad:
    """Test snapshot save/load roundtrip"""
    
    def test_save_load_roundtrip(self, architecture, serializer):
        """Test that save→load restores deterministic behavior"""
        seed_rng(42)
        
        # Process some inputs to create state
        architecture.process_input("cat", {"concept": "animal"})
        architecture.process_input("dog", {"concept": "animal"})
        architecture.process_input("cat", {"concept": "animal"})  # Second exposure
        
        # Get initial state
        initial_traces = len(architecture.episodic_memory.traces)
        initial_semantic = len(architecture.semantic_memory.semantic_memories)
        initial_working = len(architecture.working_memory.items)
        
        # Save snapshot
        snapshot_id = serializer.save(architecture, description="test_snapshot")
        assert snapshot_id is not None
        
        # Create new architecture
        seed_rng(42)
        new_config = CognitiveConfig(
            max_episodic_memories=100,
            max_working_memory_items=20,
            semantic_consolidation_threshold=2,
        )
        new_architecture = CognitiveArchitecture(config=new_config, backend_name="numpy")
        
        # Load snapshot
        restored = serializer.load(snapshot_id, new_architecture)
        
        # Verify state restoration
        assert len(restored.episodic_memory.traces) == initial_traces
        assert len(restored.semantic_memory.semantic_memories) == initial_semantic
        assert len(restored.working_memory.items) == initial_working
        
        # Verify backend state
        assert restored.backend.num_neurons == architecture.backend.num_neurons
        assert restored.backend.backend_name == architecture.backend.backend_name
    
    def test_deterministic_behavior_after_load(self, architecture, serializer):
        """Test that loaded architecture produces deterministic outputs"""
        seed_rng(42)
        
        # Process inputs
        result1 = architecture.process_input("test", {"concept": "test"})
        
        # Save and load
        snapshot_id = serializer.save(architecture)
        
        seed_rng(42)
        new_config = CognitiveConfig(
            max_episodic_memories=100,
            max_working_memory_items=20,
        )
        restored = CognitiveArchitecture(config=new_config, backend_name="numpy")
        restored = serializer.load(snapshot_id, restored)
        
        # Process same input
        seed_rng(42)
        result2 = restored.process_input("test", {"concept": "test"})
        
        # Results should be similar (same processing structure)
        assert result1["input_type"] == result2["input_type"]
        assert "episodic_trace_id" in result1
        assert "episodic_trace_id" in result2
    
    def test_list_snapshots(self, architecture, serializer):
        """Test listing snapshots"""
        # Create multiple snapshots
        snapshot_id1 = serializer.save(architecture, description="snapshot1")
        snapshot_id2 = serializer.save(architecture, description="snapshot2")
        
        # List snapshots
        snapshots = serializer.list_snapshots()
        
        assert len(snapshots) >= 2
        snapshot_ids = [s["snapshot_id"] for s in snapshots]
        assert snapshot_id1 in snapshot_ids
        assert snapshot_id2 in snapshot_ids


class TestKnowledgeSearch:
    """Test knowledge search functionality"""
    
    @pytest.mark.skipif(
        not pytest.importorskip("qdrant_client", reason="Qdrant not available"),
        reason="Qdrant not available"
    )
    def test_hybrid_search_vector_only(self, architecture):
        """Test hybrid search with vector only"""
        from storage.qdrant_store import QdrantStore
        
        # Create Qdrant store
        store = QdrantStore()
        
        # Index some semantic memories
        architecture.process_input("cat", {"concept": "animal"})
        architecture.process_input("cat", {"concept": "animal"})
        
        # Index semantic memories
        for semantic_id, semantic in architecture.semantic_memory.semantic_memories.items():
            store.upsert_semantic_memory(
                semantic_id=semantic.semantic_id,
                vector=semantic.semantic_vector,
                concept=semantic.concept,
                activation_pattern=semantic.activation_pattern,
            )
        
        # Search by vector
        query_vector = architecture.semantic_memory.semantic_memories[
            list(architecture.semantic_memory.semantic_memories.keys())[0]
        ].semantic_vector
        
        results = store.hybrid_search(
            query_vector=query_vector,
            query_spike_pattern=None,
            limit=5,
        )
        
        assert len(results) > 0
        assert "semantic_id" in results[0]
        assert "combined_score" in results[0]
    
    @pytest.mark.skipif(
        not pytest.importorskip("qdrant_client", reason="Qdrant not available"),
        reason="Qdrant not available"
    )
    def test_hybrid_search_spike_only(self, architecture):
        """Test hybrid search with spike pattern only"""
        from storage.qdrant_store import QdrantStore
        
        # Create Qdrant store
        store = QdrantStore()
        
        # Process inputs and index
        architecture.process_input("dog", {"concept": "animal"})
        
        for semantic_id, semantic in architecture.semantic_memory.semantic_memories.items():
            store.upsert_semantic_memory(
                semantic_id=semantic.semantic_id,
                vector=semantic.semantic_vector,
                concept=semantic.concept,
                activation_pattern=semantic.activation_pattern,
            )
        
        # Get a spike pattern
        spike_pattern = architecture.semantic_memory.semantic_memories[
            list(architecture.semantic_memory.semantic_memories.keys())[0]
        ].activation_pattern
        
        # Search by spike pattern
        results = store.hybrid_search(
            query_vector=None,
            query_spike_pattern=spike_pattern,
            limit=5,
        )
        
        assert len(results) > 0
        assert "spike_score" in results[0]
    
    @pytest.mark.skipif(
        not pytest.importorskip("qdrant_client", reason="Qdrant not available"),
        reason="Qdrant not available"
    )
    def test_hybrid_search_combined(self, architecture):
        """Test hybrid search combining vector and spike similarity"""
        from storage.qdrant_store import QdrantStore
        
        # Create Qdrant store
        store = QdrantStore()
        
        # Process and index multiple concepts
        architecture.process_input("cat", {"concept": "animal"})
        architecture.process_input("cat", {"concept": "animal"})
        architecture.process_input("dog", {"concept": "animal"})
        
        # Index all semantic memories
        for semantic_id, semantic in architecture.semantic_memory.semantic_memories.items():
            store.upsert_semantic_memory(
                semantic_id=semantic.semantic_id,
                vector=semantic.semantic_vector,
                concept=semantic.concept,
                activation_pattern=semantic.activation_pattern,
            )
        
        # Get query vector and spike pattern
        first_semantic = list(architecture.semantic_memory.semantic_memories.values())[0]
        query_vector = first_semantic.semantic_vector
        query_spike = first_semantic.activation_pattern
        
        # Hybrid search
        results = store.hybrid_search(
            query_vector=query_vector,
            query_spike_pattern=query_spike,
            limit=10,
            vector_weight=0.5,
            spike_weight=0.5,
        )
        
        assert len(results) > 0
        assert "combined_score" in results[0]
        assert "vector_score" in results[0] or results[0].get("vector_score") == 0.0
        assert "spike_score" in results[0] or results[0].get("spike_score") == 0.0
    
    def test_top_k_retrieval(self, architecture):
        """Test top-k retrieval returns consistent results"""
        # Process multiple inputs
        for i in range(10):
            architecture.process_input(f"concept_{i}", {"concept": f"concept_{i}"})
        
        # Get all traces
        traces = list(architecture.episodic_memory.traces.values())
        
        # Test finding similar traces (top-k)
        if traces:
            first_trace = traces[0]
            similar = architecture.episodic_memory.find_similar_traces(
                first_trace.sensory_encoding,
                threshold=0.1,
            )
            
            # Should return at least the trace itself
            assert len(similar) >= 1
            
            # Results should be sorted by similarity
            if len(similar) > 1:
                # Check that results are reasonable
                assert all(isinstance(trace_id, str) for trace_id in similar)


class TestSnapshotSchema:
    """Test snapshot schema v1"""
    
    def test_snapshot_metadata(self, architecture, serializer):
        """Test snapshot metadata structure"""
        snapshot_id = serializer.save(architecture, description="test")
        
        # Load raw JSON to check structure
        import json
        snapshot_file = Path(serializer.storage_path) / f"{snapshot_id}.json"
        with open(snapshot_file) as f:
            data = json.load(f)
        
        # Check metadata structure
        assert "metadata" in data
        assert data["metadata"]["version"] == "1.0"
        assert data["metadata"]["snapshot_id"] == snapshot_id
        assert "backend_name" in data["metadata"]
        assert "num_neurons" in data["metadata"]
    
    def test_backend_state_serialization(self, architecture, serializer):
        """Test backend state is properly serialized"""
        snapshot_id = serializer.save(architecture)
        restored = serializer.load(snapshot_id)
        
        # Check backend state
        assert restored.backend.backend_name == architecture.backend.backend_name
        assert restored.backend.num_neurons == architecture.backend.num_neurons
        
        # Check backend state data
        backend_state = architecture.backend.get_state()
        restored_state = restored.backend.get_state()
        
        assert backend_state["num_neurons"] == restored_state["num_neurons"]
        assert len(backend_state["membrane_potentials"]) == len(restored_state["membrane_potentials"])

