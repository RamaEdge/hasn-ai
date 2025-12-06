#!/usr/bin/env python3
"""
Cognitive Brain Network - Enhanced brain-inspired architecture with inference capabilities
Implements episodic memory, associative learning, and inference mechanisms
"""

import time
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Dict, List, Set, Tuple

import numpy as np

from .simplified_brain_network import (
    NetworkConfig,
    SimpleBrainNetwork,
)


@dataclass
class EpisodicMemory:
    """Represents a single episodic memory with context"""

    memory_id: str
    pattern: Dict[int, bool]
    context: Dict[str, any]
    timestamp: float
    activation_strength: float
    associations: Set[str]  # IDs of related memories
    consolidation_level: float  # How well-established this memory is

    def __post_init__(self):
        if self.associations is None:
            self.associations = set()


@dataclass
class CognitiveConfig(NetworkConfig):
    """Configuration for cognitive brain network"""

    # Memory system parameters
    max_episodic_memories: int = 1000
    memory_decay_rate: float = 0.001
    consolidation_threshold: float = 0.7
    association_strength_threshold: float = 0.3

    # Inference parameters
    inference_activation_threshold: float = 0.5
    max_inference_depth: int = 3
    temporal_association_window: float = 10.0  # seconds

    # Memory consolidation
    consolidation_probability: float = 0.1
    memory_replay_probability: float = 0.05
    
    # Learning weight combinations (configurable)
    association_weight_pattern: float = 0.4
    association_weight_temporal: float = 0.3
    association_weight_context: float = 0.3
    
    # Relevance calculation weights (configurable)
    relevance_weight_pattern: float = 0.3
    relevance_weight_context: float = 0.5
    relevance_weight_memory: float = 0.2
    min_relevance_threshold: float = 0.05


class CognitiveBrainNetwork(SimpleBrainNetwork):
    """
    Enhanced brain network with cognitive capabilities built on a spiking neural core:
    - Episodic memory storage
    - Associative learning
    - Inference generation
    - Memory consolidation

    Underlying spiking dynamics come from SimpleBrainNetwork/ SimpleSpikingNeuron
    (integrate-and-fire style update and Hebbian learning), ensuring cognitive
    features operate on top of a spiking neural substrate.
    """

    def __init__(
        self,
        num_neurons: int,
        connectivity_prob: float = 0.1,
        config: CognitiveConfig = None,
    ):
        # Initialize base network
        super().__init__(num_neurons, connectivity_prob, config or CognitiveConfig())

        # Cognitive systems
        self.episodic_memories: Dict[str, EpisodicMemory] = {}
        self.memory_associations: Dict[str, Dict[str, float]] = defaultdict(dict)
        self.inference_cache: Dict[str, List[str]] = {}
        self.recent_activations: deque = deque(maxlen=100)

        # Memory consolidation system
        self.consolidation_queue: List[str] = []
        self.memory_access_counts: Dict[str, int] = defaultdict(int)

        print(f"🧠 CognitiveBrainNetwork initialized with {num_neurons} neurons")
        print(f"   📚 Max episodic memories: {self.config.max_episodic_memories}")
        print(f"   🔗 Association threshold: {self.config.association_strength_threshold}")
        print(f"   🎯 Inference depth: {self.config.max_inference_depth}")

    def store_episodic_memory(self, pattern: Dict[int, bool], context: Dict[str, any]) -> str:
        """Store a new episodic memory with context"""
        memory_id = f"memory_{int(time.time() * 1000)}_{len(self.episodic_memories)}"

        # Create episodic memory
        memory = EpisodicMemory(
            memory_id=memory_id,
            pattern=pattern.copy(),
            context=context.copy(),
            timestamp=time.time(),
            activation_strength=1.0,
            associations=set(),
            consolidation_level=0.0,
        )

        # Store memory
        self.episodic_memories[memory_id] = memory
        self.memory_access_counts[memory_id] = 1

        # Find associations with existing memories
        self._find_memory_associations(memory_id)

        # Add to consolidation queue
        self.consolidation_queue.append(memory_id)

        # Manage memory capacity
        self._manage_memory_capacity()

        print(f"📚 Stored episodic memory: {memory_id}")
        print(f"   Context: {context}")
        print(f"   Associations found: {len(memory.associations)}")

        return memory_id

    def _find_memory_associations(self, new_memory_id: str):
        """Find associations between new memory and existing memories"""
        new_memory = self.episodic_memories[new_memory_id]

        for existing_id, existing_memory in self.episodic_memories.items():
            if existing_id == new_memory_id:
                continue

            # Calculate pattern similarity
            pattern_similarity = self._calculate_pattern_similarity(
                new_memory.pattern, existing_memory.pattern
            )

            # Calculate temporal proximity
            time_diff = abs(new_memory.timestamp - existing_memory.timestamp)
            temporal_proximity = max(0, 1 - time_diff / self.config.temporal_association_window)

            # Calculate context similarity
            context_similarity = self._calculate_context_similarity(
                new_memory.context, existing_memory.context
            )

            # Overall association strength (configurable weights)
            # Normalize weights to sum to 1.0
            total_weight = (
                self.config.association_weight_pattern +
                self.config.association_weight_temporal +
                self.config.association_weight_context
            )
            if total_weight > 0:
                weight_pattern = self.config.association_weight_pattern / total_weight
                weight_temporal = self.config.association_weight_temporal / total_weight
                weight_context = self.config.association_weight_context / total_weight
            else:
                weight_pattern = 0.4
                weight_temporal = 0.3
                weight_context = 0.3
            
            association_strength = (
                weight_pattern * pattern_similarity +
                weight_temporal * temporal_proximity +
                weight_context * context_similarity
            )

            # Create bidirectional association if strong enough
            if association_strength > self.config.association_strength_threshold:
                new_memory.associations.add(existing_id)
                existing_memory.associations.add(new_memory_id)

                # Store association strength
                self.memory_associations[new_memory_id][existing_id] = association_strength
                self.memory_associations[existing_id][new_memory_id] = association_strength

                print(
                    f"🔗 Association created: {new_memory_id} ↔ {existing_id} (strength: {association_strength:.3f})"
                )

    def _calculate_pattern_similarity(
        self, pattern1: Dict[int, bool], pattern2: Dict[int, bool]
    ) -> float:
        """Calculate similarity between two neural patterns"""
        if not pattern1 or not pattern2:
            return 0.0

        # Convert to sets for comparison
        active1 = set(k for k, v in pattern1.items() if v)
        active2 = set(k for k, v in pattern2.items() if v)

        if not active1 and not active2:
            return 1.0
        if not active1 or not active2:
            return 0.0

        # Jaccard similarity
        intersection = len(active1.intersection(active2))
        union = len(active1.union(active2))

        return intersection / union if union > 0 else 0.0

    def _calculate_context_similarity(self, context1: Dict, context2: Dict) -> float:
        """Calculate similarity between two contexts"""
        if not context1 or not context2:
            return 0.0

        # Check for exact key-value matches
        shared_keys = set(context1.keys()).intersection(set(context2.keys()))
        if not shared_keys:
            return 0.0

        exact_matches = sum(1 for key in shared_keys if context1[key] == context2[key])

        # Check for partial matches in list/array values (like 'properties')
        partial_matches = 0
        for key in shared_keys:
            val1, val2 = context1[key], context2[key]
            if isinstance(val1, list) and isinstance(val2, list):
                # Calculate overlap for lists
                overlap = len(set(val1).intersection(set(val2)))
                if overlap > 0:
                    partial_matches += overlap / max(len(val1), len(val2))
            elif str(val1).lower() in str(val2).lower() or str(val2).lower() in str(val1).lower():
                # Partial string match
                partial_matches += 0.5

        # Combine exact and partial matches
        total_similarity = (exact_matches + partial_matches) / len(shared_keys)
        return min(1.0, total_similarity)

    def generate_inferences(
        self, query_pattern: Dict[int, bool], query_context: Dict[str, any] = None
    ) -> List[Dict]:
        """Generate inferences based on query pattern and associated memories"""
        if query_context is None:
            query_context = {}

        print("\n🎯 Generating inferences for query...")

        # Find relevant memories
        relevant_memories = self._find_relevant_memories(query_pattern, query_context)

        if not relevant_memories:
            print("   No relevant memories found for inference")
            return []

        # Generate inferences through memory associations
        inferences = []
        explored_memories = set()

        for memory_id, relevance in relevant_memories[:5]:  # Top 5 relevant memories
            inference_chain = self._explore_memory_associations(
                memory_id, explored_memories, depth=0
            )

            if inference_chain:
                inferences.append(
                    {
                        "source_memory": memory_id,
                        "relevance": relevance,
                        "inference_chain": inference_chain,
                        "confidence": self._calculate_inference_confidence(inference_chain),
                    }
                )

        # Sort by confidence
        inferences.sort(key=lambda x: x["confidence"], reverse=True)

        print(f"   Generated {len(inferences)} inferences")
        for i, inf in enumerate(inferences[:3]):  # Show top 3
            print(
                f"   {i+1}. Confidence: {inf['confidence']:.3f}, Chain length: {len(inf['inference_chain'])}"
            )

        return inferences

    def _find_relevant_memories(
        self, pattern: Dict[int, bool], context: Dict[str, any]
    ) -> List[Tuple[str, float]]:
        """Find memories relevant to the query pattern and context"""
        relevance_scores = []

        for memory_id, memory in self.episodic_memories.items():
            # Calculate pattern similarity
            pattern_sim = self._calculate_pattern_similarity(pattern, memory.pattern)

            # Calculate context similarity
            context_sim = self._calculate_context_similarity(context, memory.context)

            # Consider activation strength and consolidation
            memory_strength = memory.activation_strength * (1 + memory.consolidation_level)

            # Overall relevance (configurable weights)
            # Normalize weights to sum to 1.0
            total_weight = (
                self.config.relevance_weight_pattern +
                self.config.relevance_weight_context +
                self.config.relevance_weight_memory
            )
            if total_weight > 0:
                weight_pattern = self.config.relevance_weight_pattern / total_weight
                weight_context = self.config.relevance_weight_context / total_weight
                weight_memory = self.config.relevance_weight_memory / total_weight
            else:
                weight_pattern = 0.3
                weight_context = 0.5
                weight_memory = 0.2
            
            relevance = (
                weight_pattern * pattern_sim +
                weight_context * context_sim +
                weight_memory * memory_strength
            )

            # Use configurable minimum relevance threshold
            if relevance > self.config.min_relevance_threshold:
                relevance_scores.append((memory_id, relevance))
                print(
                    f"     Memory {memory_id}: pattern_sim={pattern_sim:.3f}, context_sim={context_sim:.3f}, relevance={relevance:.3f}"
                )

        # Sort by relevance
        relevance_scores.sort(key=lambda x: x[1], reverse=True)
        return relevance_scores

    def _explore_memory_associations(
        self, memory_id: str, explored: Set[str], depth: int
    ) -> List[str]:
        """Explore memory associations to build inference chains"""
        if depth >= self.config.max_inference_depth or memory_id in explored:
            return []

        explored.add(memory_id)
        inference_chain = [memory_id]

        # Get associated memories
        memory = self.episodic_memories.get(memory_id)
        if not memory:
            return inference_chain

        # Find strongest associations
        associations = []
        for assoc_id in memory.associations:
            if assoc_id not in explored and assoc_id in self.memory_associations[memory_id]:
                strength = self.memory_associations[memory_id][assoc_id]
                associations.append((assoc_id, strength))

        # Sort by association strength
        associations.sort(key=lambda x: x[1], reverse=True)

        # Explore top associations
        for assoc_id, strength in associations[:2]:  # Top 2 associations
            if strength > self.config.association_strength_threshold:
                sub_chain = self._explore_memory_associations(assoc_id, explored, depth + 1)
                if sub_chain:
                    inference_chain.extend(sub_chain)

        return inference_chain

    def _calculate_inference_confidence(self, inference_chain: List[str]) -> float:
        """Calculate confidence in an inference chain"""
        if len(inference_chain) <= 1:
            return 0.0

        total_strength = 0.0
        connections = 0

        for i in range(len(inference_chain) - 1):
            curr_id = inference_chain[i]
            next_id = inference_chain[i + 1]

            if curr_id in self.memory_associations and next_id in self.memory_associations[curr_id]:
                total_strength += self.memory_associations[curr_id][next_id]
                connections += 1

        if connections == 0:
            return 0.0

        # Average association strength, penalized by chain length
        avg_strength = total_strength / connections
        length_penalty = 1.0 / (1.0 + 0.2 * len(inference_chain))

        return avg_strength * length_penalty

    def consolidate_memories(self):
        """Consolidate important memories through replay and strengthening"""
        if not self.consolidation_queue:
            return

        print(f"\n🔄 Consolidating {len(self.consolidation_queue)} memories...")

        consolidated_count = 0
        for memory_id in self.consolidation_queue[:]:
            if np.random.random() < self.config.consolidation_probability:
                self._consolidate_single_memory(memory_id)
                consolidated_count += 1

        # Clear processed memories from queue
        self.consolidation_queue = []

        print(f"   ✅ Consolidated {consolidated_count} memories")

    def _consolidate_single_memory(self, memory_id: str):
        """Consolidate a single memory"""
        memory = self.episodic_memories.get(memory_id)
        if not memory:
            return

        # Increase consolidation level based on access count and associations
        access_factor = min(1.0, self.memory_access_counts[memory_id] / 10.0)
        association_factor = min(1.0, len(memory.associations) / 5.0)

        consolidation_increase = 0.1 * (access_factor + association_factor)
        memory.consolidation_level = min(1.0, memory.consolidation_level + consolidation_increase)

        # Strengthen associated connections
        for assoc_id in memory.associations:
            if assoc_id in self.memory_associations[memory_id]:
                current_strength = self.memory_associations[memory_id][assoc_id]
                new_strength = min(1.0, current_strength * 1.1)
                self.memory_associations[memory_id][assoc_id] = new_strength
                self.memory_associations[assoc_id][memory_id] = new_strength

        print(f"   🔄 Consolidated {memory_id}: level={memory.consolidation_level:.3f}")

    def _manage_memory_capacity(self):
        """Manage memory capacity by removing weak memories"""
        if len(self.episodic_memories) <= self.config.max_episodic_memories:
            return

        # Find weakest memories (low consolidation + low access count)
        memory_scores = []
        for memory_id, memory in self.episodic_memories.items():
            access_count = self.memory_access_counts[memory_id]
            score = (
                memory.consolidation_level + 0.1 * access_count + 0.05 * len(memory.associations)
            )
            memory_scores.append((memory_id, score))

        # Sort by score (weakest first)
        memory_scores.sort(key=lambda x: x[1])

        # Remove weakest memories
        memories_to_remove = len(self.episodic_memories) - self.config.max_episodic_memories + 10
        for memory_id, _ in memory_scores[:memories_to_remove]:
            self._remove_memory(memory_id)

        print(f"🗑️  Removed {memories_to_remove} weak memories to maintain capacity")

    def _remove_memory(self, memory_id: str):
        """Remove a memory and clean up its associations"""
        if memory_id not in self.episodic_memories:
            return

        memory = self.episodic_memories[memory_id]

        # Remove associations
        for assoc_id in memory.associations:
            if assoc_id in self.episodic_memories:
                self.episodic_memories[assoc_id].associations.discard(memory_id)
            if assoc_id in self.memory_associations:
                self.memory_associations[assoc_id].pop(memory_id, None)

        # Clean up
        del self.episodic_memories[memory_id]
        del self.memory_associations[memory_id]
        self.memory_access_counts.pop(memory_id, None)

    def step_with_cognition(
        self, external_input: Dict[int, bool] = None, context: Dict[str, any] = None
    ) -> Dict:
        """Enhanced step function with cognitive processing"""
        # Regular network step
        spikes = self.step(external_input)

        # Store as episodic memory if we have context (always store when context provided)
        spike_count = sum(spikes.values())
        if context:  # Store memory whenever context is provided
            memory_id = self.store_episodic_memory(external_input or spikes, context)
            print(
                f"   📚 Stored memory: {context.get('concept', context.get('activity', 'unknown'))} (spikes: {spike_count})"
            )
        else:
            memory_id = None

        # Periodic memory consolidation
        if np.random.random() < self.config.consolidation_probability:
            self.consolidate_memories()

        # Generate inferences if requested in context
        inferences = []
        if context and context.get("generate_inferences", False):
            inferences = self.generate_inferences(spikes, context)

        return {
            "spikes": spikes,
            "spike_count": spike_count,
            "memory_id": memory_id,
            "inferences": inferences,
            "total_memories": len(self.episodic_memories),
            "consolidation_queue_size": len(self.consolidation_queue),
        }

    def get_cognitive_state(self) -> Dict:
        """Get current cognitive state of the network"""
        # Memory statistics
        consolidation_levels = [m.consolidation_level for m in self.episodic_memories.values()]
        association_counts = [len(m.associations) for m in self.episodic_memories.values()]

        return {
            "total_memories": len(self.episodic_memories),
            "avg_consolidation_level": (
                np.mean(consolidation_levels) if consolidation_levels else 0
            ),
            "avg_associations_per_memory": (
                np.mean(association_counts) if association_counts else 0
            ),
            "total_associations": sum(len(assocs) for assocs in self.memory_associations.values())
            // 2,
            "consolidation_queue_size": len(self.consolidation_queue),
            "memory_capacity_usage": len(self.episodic_memories)
            / self.config.max_episodic_memories,
        }


# Note: The interactive demo for cognitive inference has been moved to
# examples/cognitive_inference_demo.py to keep core modules free of CLI/demo code.
