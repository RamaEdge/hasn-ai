#!/usr/bin/env python3
"""
Tests for Cognitive Layers - Comprehensive testing of cognitive architecture
Tests TTL eviction, consolidation, and all cognitive layer functionality
"""

import os
import sys
import time
from unittest.mock import Mock

import pytest

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from core.cognitive_architecture import CognitiveArchitecture
from core.cognitive_layers import (
    AssociativeLayer,
    EpisodicMemoryLayer,
    ExecutiveLayer,
    SemanticMemoryLayer,
    SensoryLayer,
    WorkingMemoryLayer,
)
from core.cognitive_models import CognitiveConfig, EpisodeTrace


class TestSensoryLayer:
    """Test sensory encoding layer"""

    def setup_method(self):
        self.config = CognitiveConfig()
        self.backend = Mock()
        self.backend.num_neurons = 100
        self.sensory_layer = SensoryLayer(self.config, self.backend)

    def test_encode_text_to_spikes(self):
        """Test text to spike encoding"""
        text = "hello"
        spike_pattern = self.sensory_layer.encode_text_to_spikes(text)

        assert isinstance(spike_pattern, dict)
        assert len(spike_pattern) == self.backend.num_neurons
        assert all(isinstance(spike, bool) for spike in spike_pattern.values())

        # Should have some spikes for non-empty text
        assert sum(spike_pattern.values()) > 0

    def test_encode_embedding_to_spikes(self):
        """Test embedding to spike encoding"""
        embedding = [0.1, 0.5, -0.3, 0.8, -0.2]
        spike_pattern = self.sensory_layer.encode_embedding_to_spikes(embedding)

        assert isinstance(spike_pattern, dict)
        assert len(spike_pattern) == self.backend.num_neurons
        assert all(isinstance(spike, bool) for spike in spike_pattern.values())

    def test_encoding_cache(self):
        """Test that encoding is cached"""
        text = "test"

        # First encoding
        pattern1 = self.sensory_layer.encode_text_to_spikes(text)

        # Second encoding should be cached
        pattern2 = self.sensory_layer.encode_text_to_spikes(text)

        assert pattern1 == pattern2
        assert text in self.sensory_layer.encoding_cache


class TestAssociativeLayer:
    """Test associative memory layer"""

    def setup_method(self):
        self.config = CognitiveConfig()
        self.backend = Mock()
        self.backend.num_neurons = 100
        self.associative_layer = AssociativeLayer(self.config, self.backend)

    def test_update_associations(self):
        """Test Hebbian learning updates"""
        spike_pattern = {0: True, 1: True, 2: False, 3: True}
        timestamp = time.time()

        initial_strength = self.associative_layer.association_matrix[0, 1]

        self.associative_layer.update_associations(spike_pattern, timestamp)

        # Should strengthen connections between co-active neurons
        assert self.associative_layer.association_matrix[0, 1] > initial_strength
        assert self.associative_layer.association_matrix[0, 3] > initial_strength
        assert self.associative_layer.association_matrix[1, 3] > initial_strength

    def test_get_associated_neurons(self):
        """Test getting associated neurons"""
        # Set up some associations
        self.associative_layer.association_matrix[0, 1] = 0.5
        self.associative_layer.association_matrix[0, 2] = 0.2
        self.associative_layer.association_matrix[0, 3] = 0.8

        associations = self.associative_layer.get_associated_neurons(0, threshold=0.3)

        assert 1 in associations
        assert 3 in associations
        assert 2 not in associations  # Below threshold

    def test_association_decay(self):
        """Test that associations decay over time"""
        # Set up strong association
        self.associative_layer.association_matrix[0, 1] = 1.0

        # Apply decay
        self.associative_layer._decay_associations()

        # Should be reduced
        assert self.associative_layer.association_matrix[0, 1] < 1.0


class TestWorkingMemoryLayer:
    """Test working memory with TTL eviction"""

    def setup_method(self):
        self.config = CognitiveConfig(max_working_memory_items=3, working_memory_ttl=1.0)
        self.working_memory = WorkingMemoryLayer(self.config)

    def test_add_and_get_item(self):
        """Test adding and retrieving items"""
        content = "test content"
        item_id = self.working_memory.add_item(content)

        assert item_id in self.working_memory.items
        retrieved = self.working_memory.get_item(item_id)
        assert retrieved == content

    def test_ttl_eviction(self):
        """Test TTL-based eviction"""
        # Add item with short TTL
        item_id = self.working_memory.add_item("test", ttl=0.1)

        # Should be available immediately
        assert self.working_memory.get_item(item_id) is not None

        # Wait for expiration
        time.sleep(0.2)

        # Should be expired
        assert self.working_memory.get_item(item_id) is None
        assert item_id not in self.working_memory.items

    def test_capacity_eviction(self):
        """Test capacity-based eviction (LRU)"""
        # Add items up to capacity
        item_ids = []
        for i in range(4):  # More than max capacity (3)
            item_id = self.working_memory.add_item(f"content_{i}")
            item_ids.append(item_id)

        # Should have evicted oldest item
        assert len(self.working_memory.items) == 3

        # First item should be evicted
        assert self.working_memory.get_item(item_ids[0]) is None

        # Last items should still be available
        assert self.working_memory.get_item(item_ids[-1]) is not None

    def test_access_order_update(self):
        """Test that access order is updated on retrieval"""
        # Add items
        item1 = self.working_memory.add_item("content1")
        item2 = self.working_memory.add_item("content2")
        item3 = self.working_memory.add_item("content3")

        # Access first item (should move to end)
        self.working_memory.get_item(item1)

        # Add one more item (should evict item2, not item1)
        item4 = self.working_memory.add_item("content4")

        assert self.working_memory.get_item(item1) is not None
        assert self.working_memory.get_item(item2) is None
        assert self.working_memory.get_item(item3) is not None
        assert self.working_memory.get_item(item4) is not None

    def test_cleanup_expired(self):
        """Test cleanup of expired items"""
        # Add items with different TTLs
        item1 = self.working_memory.add_item("content1", ttl=0.1)
        item2 = self.working_memory.add_item("content2", ttl=1.0)

        # Wait for first item to expire
        time.sleep(0.2)

        # Cleanup
        self.working_memory.cleanup_expired()

        assert item1 not in self.working_memory.items
        assert item2 in self.working_memory.items


class TestEpisodicMemoryLayer:
    """Test episodic memory layer"""

    def setup_method(self):
        self.config = CognitiveConfig(max_episodic_memories=5)
        self.episodic_memory = EpisodicMemoryLayer(self.config)

    def test_store_and_retrieve_trace(self):
        """Test storing and retrieving episode traces"""
        trace = EpisodeTrace(
            trace_id="test_trace",
            sensory_input="test input",
            sensory_encoding={0: True, 1: False, 2: True},
        )

        trace_id = self.episodic_memory.store_trace(trace)
        assert trace_id == "test_trace"

        retrieved = self.episodic_memory.retrieve_trace(trace_id)
        assert retrieved is not None
        assert retrieved.trace_id == "test_trace"
        assert retrieved.access_count == 2  # 1 from creation, 1 from retrieval

    def test_find_similar_traces(self):
        """Test finding similar traces"""
        # Add traces with different patterns
        trace1 = EpisodeTrace(
            trace_id="trace1", sensory_input="input1", sensory_encoding={0: True, 1: True, 2: False}
        )
        trace2 = EpisodeTrace(
            trace_id="trace2",
            sensory_input="input2",
            sensory_encoding={0: True, 1: True, 2: True},  # Similar to trace1
        )
        trace3 = EpisodeTrace(
            trace_id="trace3",
            sensory_input="input3",
            sensory_encoding={0: False, 1: False, 2: False},  # Different
        )

        self.episodic_memory.store_trace(trace1)
        self.episodic_memory.store_trace(trace2)
        self.episodic_memory.store_trace(trace3)

        # Find traces similar to trace1
        similar = self.episodic_memory.find_similar_traces(trace1.sensory_encoding, threshold=0.5)

        assert "trace2" in similar  # High similarity
        assert "trace3" not in similar  # Low similarity

    def test_capacity_management(self):
        """Test that capacity is managed by removing oldest traces"""
        # Add traces up to capacity
        for i in range(7):  # More than max capacity (5)
            trace = EpisodeTrace(
                trace_id=f"trace_{i}", sensory_input=f"input_{i}", sensory_encoding={i: True}
            )
            self.episodic_memory.store_trace(trace)

        # Should have evicted oldest traces
        assert len(self.episodic_memory.traces) == 5

        # Oldest traces should be gone
        assert self.episodic_memory.retrieve_trace("trace_0") is None
        assert self.episodic_memory.retrieve_trace("trace_1") is None

        # Newest traces should still be there
        assert self.episodic_memory.retrieve_trace("trace_6") is not None


class TestSemanticMemoryLayer:
    """Test semantic memory layer"""

    def setup_method(self):
        self.config = CognitiveConfig()
        self.backend = Mock()
        self.backend.num_neurons = 100
        self.semantic_memory = SemanticMemoryLayer(self.config, self.backend)

    def test_consolidate_traces(self):
        """Test consolidating traces into semantic memory"""
        trace_ids = ["trace1", "trace2", "trace3"]
        concept = "test_concept"

        semantic_id = self.semantic_memory.consolidate_traces(trace_ids, concept)

        assert semantic_id is not None
        assert semantic_id in self.semantic_memory.semantic_memories

        semantic_memory = self.semantic_memory.semantic_memories[semantic_id]
        assert semantic_memory.concept == concept
        assert semantic_memory.consolidated_traces == set(trace_ids)
        assert semantic_memory.consolidated_count == 3

    def test_find_semantic_memory(self):
        """Test finding semantic memory by concept"""
        trace_ids = ["trace1", "trace2"]
        concept = "cat"

        semantic_id = self.semantic_memory.consolidate_traces(trace_ids, concept)

        found = self.semantic_memory.find_semantic_memory(concept)
        assert found is not None
        assert found.concept == concept
        assert found.semantic_id == semantic_id


class TestExecutiveLayer:
    """Test executive layer arbitration"""

    def setup_method(self):
        self.config = CognitiveConfig()
        self.executive_layer = ExecutiveLayer(self.config)

    def test_arbitration(self):
        """Test arbitration logic"""
        # Test explicit consolidation request
        context = {"consolidation_request": True}
        decision = self.executive_layer.arbitrate(context)
        assert decision == "consolidate"

        # Test explicit recall request
        context = {"recall_request": True}
        decision = self.executive_layer.arbitrate(context)
        assert decision == "recall"

        # Test default arbitration (should be either recall or consolidate)
        context = {}
        decision = self.executive_layer.arbitrate(context)
        assert decision in ["recall", "consolidate"]

    def test_consolidation_decision(self):
        """Test consolidation decision logic"""
        # Should consolidate if access count >= threshold
        assert self.executive_layer.should_consolidate("trace1", 5)  # threshold is 5
        assert not self.executive_layer.should_consolidate("trace2", 3)

    def test_consolidation_priority(self):
        """Test consolidation priority calculation"""
        priority = self.executive_layer.get_consolidation_priority("trace1", 10, 0.8)
        assert priority > 0

        # Higher access count should give higher priority
        priority1 = self.executive_layer.get_consolidation_priority("trace1", 5, 0.5)
        priority2 = self.executive_layer.get_consolidation_priority("trace2", 10, 0.5)
        assert priority2 > priority1


class TestCognitiveArchitecture:
    """Test complete cognitive architecture"""

    def setup_method(self):
        self.config = CognitiveConfig(
            max_working_memory_items=10,
            working_memory_ttl=1.0,
            semantic_consolidation_threshold=2,  # Lower threshold for testing
        )
        self.architecture = CognitiveArchitecture(self.config)

    def test_process_text_input(self):
        """Test processing text input through complete pipeline"""
        result = self.architecture.process_input("hello world", {"test": True})

        assert "processing_id" in result
        assert result["input_type"] == "text"
        assert "spike_pattern" in result
        assert "working_memory_id" in result
        assert "episodic_trace_id" in result
        assert result["arbitration_decision"] in ["recall", "consolidate"]
        assert result["processing_time"] > 0

    def test_process_embedding_input(self):
        """Test processing embedding input"""
        embedding = [0.1, 0.5, -0.3, 0.8]
        result = self.architecture.process_input(embedding, {"test": True})

        assert result["input_type"] == "embedding"
        assert "spike_pattern" in result

    def test_working_memory_ttl_eviction(self):
        """Test TTL eviction in working memory"""
        # Process input
        result = self.architecture.process_input("test", {"ttl_test": True})
        working_id = result["working_memory_id"]

        # Should be available immediately
        assert self.architecture.working_memory.get_item(working_id) is not None

        # Wait for expiration
        time.sleep(1.1)

        # Should be expired
        assert self.architecture.working_memory.get_item(working_id) is None

    def test_episodic_memory_storage(self):
        """Test episodic memory storage and retrieval"""
        result = self.architecture.process_input("episodic test", {"episodic_test": True})
        trace_id = result["episodic_trace_id"]

        # Should be stored in episodic memory
        trace = self.architecture.episodic_memory.retrieve_trace(trace_id)
        assert trace is not None
        assert trace.sensory_input == "episodic test"

    def test_semantic_consolidation(self):
        """Test semantic memory consolidation after multiple exposures"""
        # Process same input multiple times to trigger consolidation
        for i in range(4):  # More than threshold (3)
            self.architecture.process_input("cat", {"exposure": i})

        # Check if consolidation occurred
        cat_semantic = self.architecture.semantic_memory.find_semantic_memory("cat")
        assert cat_semantic is not None
        assert cat_semantic.concept == "cat"

    def test_cat_example_demonstration(self):
        """Test the complete cat example demonstration"""
        demo_result = self.architecture.demonstrate_cat_example()

        assert demo_result["exposures"] == 6
        assert demo_result["consolidation_occurred"] is True
        assert demo_result["semantic_memory"] is not None
        assert len(demo_result["processing_results"]) == 6

    def test_system_status(self):
        """Test system status reporting"""
        # Process some inputs
        self.architecture.process_input("test1")
        self.architecture.process_input("test2")

        status = self.architecture.get_system_status()

        assert "timestamp" in status
        assert "backend" in status
        assert "neurons" in status
        assert status["total_activations"] >= 2
        assert "working_memory_items" in status
        assert "episodic_memories" in status
        assert "semantic_memories" in status

    def test_memory_statistics(self):
        """Test memory statistics reporting"""
        # Process some inputs
        self.architecture.process_input("stat_test1")
        self.architecture.process_input("stat_test2")

        stats = self.architecture.get_memory_statistics()

        assert "working_memory" in stats
        assert "episodic_memory" in stats
        assert "semantic_memory" in stats

        assert stats["working_memory"]["total_items"] >= 0
        assert stats["episodic_memory"]["total_traces"] >= 0
        assert stats["semantic_memory"]["total_memories"] >= 0

    def test_cleanup_expired_memories(self):
        """Test cleanup of expired memories"""
        # Process input with short TTL
        result = self.architecture.process_input("cleanup_test", {"cleanup": True})

        # Wait for expiration
        time.sleep(1.1)

        # Cleanup
        self.architecture.cleanup_expired_memories()

        # Should have cleaned up expired items
        working_id = result["working_memory_id"]
        assert self.architecture.working_memory.get_item(working_id) is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
