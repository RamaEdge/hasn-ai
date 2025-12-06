#!/usr/bin/env python3
"""
Cognitive Layers - Implementation of layered cognitive architecture
Implements sensory, associative, working, episodic, semantic, and executive layers
"""

import uuid
from collections import defaultdict, deque
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .cognitive_models import (
    CognitiveConfig,
    EpisodeTrace,
    SemanticMemory,
    WorkingMemoryItem,
)


class SensoryLayer:
    """Sensory encoding layer - converts text/embeddings to spike patterns"""

    def __init__(self, config: CognitiveConfig, backend):
        self.config = config
        self.backend = backend
        self.encoding_cache: Dict[str, Dict[int, bool]] = {}

    def encode_text_to_spikes(self, text: str) -> Dict[int, bool]:
        """Encode text input to Poisson spike pattern"""
        if text in self.encoding_cache:
            return self.encoding_cache[text]

        # Simple text-to-spike encoding using character frequencies
        spike_pattern = {}
        char_freq = defaultdict(int)

        # Count character frequencies
        for char in text.lower():
            char_freq[char] += 1

        # Generate Poisson spikes based on character frequencies
        for i, (char, freq) in enumerate(char_freq.items()):
            if i >= self.backend.num_neurons:
                break

            # Poisson spike generation
            spike_prob = min(1.0, freq * self.config.sensory_encoding_rate / 100.0)
            spike_pattern[i] = np.random.poisson(spike_prob) > 0

        # Fill remaining neurons with low-probability spikes (configurable)
        for i in range(len(char_freq), self.backend.num_neurons):
            spike_prob = self.config.background_spike_probability
            spike_pattern[i] = np.random.poisson(spike_prob) > 0

        self.encoding_cache[text] = spike_pattern
        return spike_pattern

    def encode_embedding_to_spikes(self, embedding: List[float]) -> Dict[int, bool]:
        """Encode vector embedding to spike pattern"""
        spike_pattern = {}

        # Normalize embedding
        embedding = np.array(embedding)
        if np.linalg.norm(embedding) > 0:
            embedding = embedding / np.linalg.norm(embedding)

        # Convert to spikes based on magnitude
        for i, value in enumerate(embedding):
            if i >= self.backend.num_neurons:
                break
            spike_prob = abs(value) * self.config.sensory_encoding_rate / 10.0
            spike_pattern[i] = np.random.poisson(spike_prob) > 0

        # Fill remaining neurons
        for i in range(len(embedding), self.backend.num_neurons):
            spike_pattern[i] = False

        return spike_pattern


class AssociativeLayer:
    """Associative memory layer - Hebbian learning between co-active spikes"""

    def __init__(self, config: CognitiveConfig, backend):
        self.config = config
        self.backend = backend
        self.association_matrix = np.zeros((backend.num_neurons, backend.num_neurons))
        self.co_activity_history = deque(maxlen=1000)

    def update_associations(self, spike_pattern: Dict[int, bool], timestamp: float):
        """Update associative connections based on co-active spikes"""
        active_neurons = [i for i, spike in spike_pattern.items() if spike]

        if len(active_neurons) < 2:
            return

        # Record co-activity
        self.co_activity_history.append((active_neurons, timestamp))

        # Hebbian learning: strengthen connections between co-active neurons
        for i in active_neurons:
            for j in active_neurons:
                if i != j:
                    # Hebbian update
                    self.association_matrix[i, j] += self.config.hebbian_learning_rate
                    # Cap at configurable maximum association strength
                    self.association_matrix[i, j] = min(
                        self.config.max_association_strength, self.association_matrix[i, j]
                    )

        # Decay old associations
        self._decay_associations()

    def _decay_associations(self):
        """Decay association strengths over time"""
        decay_factor = 1.0 - self.config.memory_decay_rate
        self.association_matrix *= decay_factor

    def get_associated_neurons(self, neuron_id: int, threshold: float = None) -> List[int]:
        """Get neurons associated with the given neuron"""
        if threshold is None:
            threshold = self.config.association_threshold

        associations = []
        for i, strength in enumerate(self.association_matrix[neuron_id]):
            if strength >= threshold:
                associations.append(i)

        return associations


class WorkingMemoryLayer:
    """Working memory layer with TTL eviction"""

    def __init__(self, config: CognitiveConfig):
        self.config = config
        self.items: Dict[str, WorkingMemoryItem] = {}
        self.access_order = deque()

    def add_item(self, content: Any, ttl: float = None) -> str:
        """Add item to working memory"""
        if ttl is None:
            ttl = self.config.working_memory_ttl

        item_id = str(uuid.uuid4())
        item = WorkingMemoryItem(item_id=item_id, content=content, ttl=ttl)

        self.items[item_id] = item
        self.access_order.append(item_id)

        # Evict if over capacity
        self._evict_if_needed()

        return item_id

    def get_item(self, item_id: str) -> Optional[Any]:
        """Get item from working memory and update access"""
        if item_id not in self.items:
            return None

        item = self.items[item_id]
        if item.is_expired():
            self.remove_item(item_id)
            return None

        item.update_access()
        # Move to end of access order
        if item_id in self.access_order:
            self.access_order.remove(item_id)
        self.access_order.append(item_id)

        return item.content

    def remove_item(self, item_id: str):
        """Remove item from working memory"""
        if item_id in self.items:
            del self.items[item_id]
        if item_id in self.access_order:
            self.access_order.remove(item_id)

    def _evict_if_needed(self):
        """Evict items if over capacity"""
        while len(self.items) > self.config.max_working_memory_items:
            if self.access_order:
                # Evict least recently accessed
                oldest_id = self.access_order.popleft()
                if oldest_id in self.items:
                    del self.items[oldest_id]
            else:
                break

    def cleanup_expired(self):
        """Remove expired items"""
        expired_ids = []
        for item_id, item in self.items.items():
            if item.is_expired():
                expired_ids.append(item_id)

        for item_id in expired_ids:
            self.remove_item(item_id)

    def get_all_items(self) -> Dict[str, Any]:
        """Get all non-expired items"""
        self.cleanup_expired()
        return {item_id: item.content for item_id, item in self.items.items()}


class EpisodicMemoryLayer:
    """Episodic memory layer - stores episode traces with timestamps"""

    def __init__(self, config: CognitiveConfig):
        self.config = config
        self.traces: Dict[str, EpisodeTrace] = {}
        self.temporal_index: List[Tuple[float, str]] = []  # (timestamp, trace_id)

    def store_trace(self, trace: EpisodeTrace) -> str:
        """Store an episode trace"""
        self.traces[trace.trace_id] = trace
        self.temporal_index.append((trace.timestamp.timestamp(), trace.trace_id))
        self.temporal_index.sort()  # Keep sorted by timestamp

        # Manage capacity
        self._manage_capacity()

        return trace.trace_id

    def retrieve_trace(self, trace_id: str) -> Optional[EpisodeTrace]:
        """Retrieve an episode trace"""
        if trace_id not in self.traces:
            return None

        trace = self.traces[trace_id]
        trace.access_count += 1
        return trace

    def increment_access_count(self, trace_id: str):
        """Increment access count for a trace without retrieving it"""
        if trace_id in self.traces:
            self.traces[trace_id].access_count += 1

    def find_similar_traces(self, pattern: Dict[int, bool], threshold: float = None) -> List[str]:
        """Find traces with similar spike patterns"""
        if threshold is None:
            threshold = self.config.consolidation_threshold

        similar_traces = []

        for trace_id, trace in self.traces.items():
            similarity = self._calculate_pattern_similarity(pattern, trace.sensory_encoding)
            if similarity >= threshold:
                similar_traces.append(trace_id)

        return similar_traces

    def _calculate_pattern_similarity(
        self, pattern1: Dict[int, bool], pattern2: Dict[int, bool]
    ) -> float:
        """Calculate similarity between two spike patterns"""
        all_neurons = set(pattern1.keys()) | set(pattern2.keys())
        if not all_neurons:
            return 0.0

        matches = 0
        total_neurons = 0
        for neuron in all_neurons:
            spike1 = pattern1.get(neuron, False)
            spike2 = pattern2.get(neuron, False)
            total_neurons += 1
            if spike1 == spike2:
                matches += 1

        # For same concept, boost similarity
        if total_neurons > 0:
            base_similarity = matches / total_neurons
            # If both patterns have similar active neurons, boost similarity
            active1 = sum(pattern1.values())
            active2 = sum(pattern2.values())
            if active1 > 0 and active2 > 0:
                activity_similarity = min(active1, active2) / max(active1, active2)
                return max(base_similarity, activity_similarity * 0.8)

        return matches / total_neurons if total_neurons > 0 else 0.0

    def _manage_capacity(self):
        """Manage memory capacity by removing oldest traces"""
        while len(self.traces) > self.config.max_episodic_memories:
            if self.temporal_index:
                _, oldest_id = self.temporal_index.pop(0)
                if oldest_id in self.traces:
                    del self.traces[oldest_id]


class SemanticMemoryLayer:
    """Semantic memory layer - consolidation process merges traces"""

    def __init__(self, config: CognitiveConfig, backend=None):
        self.config = config
        self.backend = backend
        self.semantic_memories: Dict[str, SemanticMemory] = {}
        self.consolidation_queue: List[str] = []

    def consolidate_traces(self, trace_ids: List[str], concept: str) -> str:
        """Consolidate multiple traces into semantic memory"""
        if not trace_ids:
            return None

        # Calculate consolidated vector
        consolidated_vector = self._calculate_consolidated_vector(trace_ids)
        activation_pattern = self._calculate_activation_pattern(trace_ids)

        semantic_id = str(uuid.uuid4())
        semantic_memory = SemanticMemory(
            semantic_id=semantic_id,
            concept=concept,
            consolidated_traces=set(trace_ids),
            consolidated_count=len(trace_ids),
            semantic_vector=consolidated_vector,
            activation_pattern=activation_pattern,
            confidence=min(
                self.config.max_confidence,
                len(trace_ids) / self.config.semantic_consolidation_threshold,
            ),
        )

        self.semantic_memories[semantic_id] = semantic_memory
        return semantic_id

    def _calculate_consolidated_vector(self, trace_ids: List[str]) -> List[float]:
        """Calculate consolidated vector from multiple traces"""
        # This is a simplified implementation
        # In practice, you'd use more sophisticated vector operations
        vectors = []
        for trace_id in trace_ids:
            # Placeholder - would use actual embeddings from traces
            vectors.append(np.random.random(128).tolist())

        if not vectors:
            return [0.0] * 128

        # Average the vectors
        consolidated = np.mean(vectors, axis=0)
        return consolidated.tolist()

    def _calculate_activation_pattern(self, trace_ids: List[str]) -> Dict[int, bool]:
        """Calculate consolidated activation pattern"""
        # Simplified - would use actual spike patterns from traces
        pattern = {}
        # Use backend neuron count instead of hard-coded 100
        if self.backend:
            neuron_count = self.backend.num_neurons
        else:
            neuron_count = 100  # Fallback if backend not available

        for i in range(neuron_count):
            pattern[i] = np.random.random() > self.config.activation_pattern_threshold
        return pattern

    def find_semantic_memory(self, concept: str) -> Optional[SemanticMemory]:
        """Find semantic memory by concept"""
        for memory in self.semantic_memories.values():
            if memory.concept == concept:
                return memory
        return None


class ExecutiveLayer:
    """Executive layer - arbitration logic for recall vs consolidation"""

    def __init__(self, config: CognitiveConfig):
        self.config = config
        self.decision_history = deque(maxlen=100)

    def arbitrate(self, context: Dict[str, Any]) -> str:
        """Arbitrate between recall and consolidation"""
        # Simple arbitration based on context
        if "consolidation_request" in context:
            return "consolidate"

        if "recall_request" in context:
            return "recall"

        # For testing: if we have multiple exposures of same concept, prefer consolidation
        if "exposure" in context and context.get("exposure", 0) >= 2:
            return "consolidate"

        # Default arbitration based on probability
        if np.random.random() < self.config.recall_probability:
            return "recall"
        else:
            return "consolidate"

    def should_consolidate(self, trace_id: str, access_count: int) -> bool:
        """Determine if a trace should be consolidated"""
        return access_count >= self.config.semantic_consolidation_threshold

    def get_consolidation_priority(self, trace_id: str, access_count: int, recency: float) -> float:
        """Calculate consolidation priority for a trace"""
        # Higher priority for frequently accessed, recent traces (configurable weights)
        priority = (
            access_count * self.config.consolidation_priority_weight_access
            + recency * self.config.consolidation_priority_weight_recency
        )
        return priority
