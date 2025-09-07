#!/usr/bin/env python3
"""
Cognitive Architecture - Main integration of all cognitive layers
Implements the complete cognitive system with sensory encoding, memory systems, and executive control
"""

import time
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set, Tuple, Any, Union
import numpy as np

from .cognitive_models import (
    CognitiveConfig,
    EpisodeTrace,
    WorkingMemoryItem,
    SemanticMemory,
    LayerState,
    CognitiveState
)
from .cognitive_layers import (
    SensoryLayer,
    AssociativeLayer,
    WorkingMemoryLayer,
    EpisodicMemoryLayer,
    SemanticMemoryLayer,
    ExecutiveLayer
)
from .backend.factory import get_backend


class CognitiveArchitecture:
    """
    Complete cognitive architecture integrating all layers:
    - Sensory encoding (text/embeddings → spikes)
    - Associative memory (Hebbian learning)
    - Working memory (TTL eviction)
    - Episodic memory (episode traces)
    - Semantic memory (consolidation)
    - Executive layer (arbitration)
    """
    
    def __init__(self, config: CognitiveConfig = None, backend_name: str = "numpy"):
        self.config = config or CognitiveConfig()
        self.backend = get_backend(backend_name, num_neurons=1000)
        
        # Initialize cognitive layers
        self.sensory_layer = SensoryLayer(self.config, self.backend)
        self.associative_layer = AssociativeLayer(self.config, self.backend)
        self.working_memory = WorkingMemoryLayer(self.config)
        self.episodic_memory = EpisodicMemoryLayer(self.config)
        self.semantic_memory = SemanticMemoryLayer(self.config)
        self.executive_layer = ExecutiveLayer(self.config)
        
        # System state
        self.current_state = CognitiveState()
        self.processing_history = []
        
        print(f"🧠 Cognitive Architecture initialized")
        print(f"   🔧 Backend: {self.backend.backend_name}")
        print(f"   🧬 Neurons: {self.backend.num_neurons}")
        print(f"   📚 Max episodic memories: {self.config.max_episodic_memories}")
        print(f"   💭 Max working memory items: {self.config.max_working_memory_items}")
    
    def process_input(self, input_data: Union[str, List[float]], context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Process input through the complete cognitive pipeline"""
        if context is None:
            context = {}
        
        start_time = time.time()
        processing_id = str(uuid.uuid4())
        
        print(f"🔄 Processing input: {processing_id}")
        
        # 1. Sensory encoding
        if isinstance(input_data, str):
            spike_pattern = self.sensory_layer.encode_text_to_spikes(input_data)
            input_type = "text"
        else:
            spike_pattern = self.sensory_layer.encode_embedding_to_spikes(input_data)
            input_type = "embedding"
        
        print(f"   🎯 Sensory encoding: {sum(spike_pattern.values())} spikes")
        
        # 2. Associative learning
        self.associative_layer.update_associations(spike_pattern, time.time())
        
        # 3. Working memory storage
        working_item_id = self.working_memory.add_item({
            "input": input_data,
            "spike_pattern": spike_pattern,
            "context": context,
            "timestamp": datetime.now()
        })
        
        # 4. Create episode trace
        trace = EpisodeTrace(
            trace_id=str(uuid.uuid4()),
            sensory_input=str(input_data) if isinstance(input_data, str) else "embedding",
            sensory_encoding=spike_pattern,
            context=context
        )
        
        # 5. Store in episodic memory
        trace_id = self.episodic_memory.store_trace(trace)
        
        # 5.5. Increment access count for similar traces (same concept)
        concept = self._extract_concept(trace)
        for existing_trace_id, existing_trace in self.episodic_memory.traces.items():
            if existing_trace_id != trace_id:
                existing_concept = self._extract_concept(existing_trace)
                if existing_concept == concept:
                    self.episodic_memory.increment_access_count(existing_trace_id)
        
        # 6. Executive arbitration
        arbitration_decision = self.executive_layer.arbitrate(context)
        
        # 7. Process based on arbitration
        result = {
            "processing_id": processing_id,
            "input_type": input_type,
            "spike_pattern": spike_pattern,
            "working_memory_id": working_item_id,
            "episodic_trace_id": trace_id,
            "arbitration_decision": arbitration_decision,
            "processing_time": time.time() - start_time
        }
        
        if arbitration_decision == "recall":
            # Find similar traces
            similar_traces = self.episodic_memory.find_similar_traces(spike_pattern)
            result["similar_traces"] = similar_traces
            result["recall_count"] = len(similar_traces)
            
        elif arbitration_decision == "consolidate":
            # Check for consolidation opportunities
            consolidation_result = self._check_consolidation_opportunities(trace_id)
            result["consolidation_result"] = consolidation_result
            
            # Debug output
            if consolidation_result.get("consolidated"):
                print(f"   🧠 Consolidation successful: {consolidation_result['concept']}")
            else:
                print(f"   ⚠️  Consolidation failed: {consolidation_result.get('reason', 'unknown')}")
        
        # Update system state
        self._update_system_state(result)
        
        print(f"   ✅ Processing complete: {result['processing_time']:.3f}s")
        print(f"   🎯 Decision: {arbitration_decision}")
        
        return result
    
    def _check_consolidation_opportunities(self, trace_id: str) -> Dict[str, Any]:
        """Check if there are opportunities for memory consolidation"""
        trace = self.episodic_memory.retrieve_trace(trace_id)
        if not trace:
            return {"consolidated": False, "reason": "trace_not_found"}
        
        # Check if trace should be consolidated
        should_consolidate = self.executive_layer.should_consolidate(
            trace_id, trace.access_count
        )
        
        if should_consolidate:
            concept = self._extract_concept(trace)
            
            # Find traces with same concept (more aggressive for same concept)
            concept_traces = []
            for other_trace_id, other_trace in self.episodic_memory.traces.items():
                if other_trace_id != trace_id:
                    other_concept = self._extract_concept(other_trace)
                    if other_concept == concept:
                        concept_traces.append(other_trace_id)
            
            # If we have enough traces of the same concept, consolidate
            if len(concept_traces) >= 1:  # Lower threshold - just need 1 other trace
                all_traces = [trace_id] + concept_traces
                
                # Consolidate into semantic memory
                semantic_id = self.semantic_memory.consolidate_traces(
                    all_traces, concept
                )
                
                return {
                    "consolidated": True,
                    "semantic_id": semantic_id,
                    "consolidated_traces": all_traces,
                    "concept": concept
                }
            
            # Fallback: find similar traces for consolidation
            similar_traces = self.episodic_memory.find_similar_traces(
                trace.sensory_encoding, threshold=0.3
            )
            
            if len(similar_traces) >= 2:
                all_traces = [trace_id] + similar_traces
                semantic_id = self.semantic_memory.consolidate_traces(
                    all_traces, concept
                )
                
                return {
                    "consolidated": True,
                    "semantic_id": semantic_id,
                    "consolidated_traces": all_traces,
                    "concept": concept
                }
        
        return {"consolidated": False, "reason": "insufficient_similarity_or_access"}
    
    def _extract_concept(self, trace: EpisodeTrace) -> str:
        """Extract concept from episode trace"""
        # Simple concept extraction - in practice, this would be more sophisticated
        if isinstance(trace.sensory_input, str):
            # Extract key words or use NLP
            words = trace.sensory_input.lower().split()
            if words:
                return words[0]  # Use first word as concept
        return "unknown_concept"
    
    def _update_system_state(self, result: Dict[str, Any]):
        """Update the overall system state"""
        self.current_state.total_activations += 1
        
        if result.get("arbitration_decision") == "recall":
            self.current_state.recall_events += 1
        elif result.get("arbitration_decision") == "consolidate":
            self.current_state.consolidation_events += 1
        
        # Update layer states
        self.current_state.layer_states["sensory"] = LayerState(
            layer_name="sensory",
            active_neurons=set(i for i, spike in result["spike_pattern"].items() if spike)
        )
        
        # Store processing history
        self.processing_history.append(result)
        if len(self.processing_history) > 1000:  # Keep last 1000 processes
            self.processing_history.pop(0)
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        return {
            "timestamp": datetime.now(),
            "backend": self.backend.backend_name,
            "neurons": self.backend.num_neurons,
            "working_memory_items": len(self.working_memory.items),
            "episodic_memories": len(self.episodic_memory.traces),
            "semantic_memories": len(self.semantic_memory.semantic_memories),
            "total_activations": self.current_state.total_activations,
            "recall_events": self.current_state.recall_events,
            "consolidation_events": self.current_state.consolidation_events,
            "processing_history_length": len(self.processing_history)
        }
    
    def cleanup_expired_memories(self):
        """Clean up expired working memory items"""
        self.working_memory.cleanup_expired()
    
    def demonstrate_cat_example(self) -> Dict[str, Any]:
        """Demonstrate the cat example: encode 'cat' → spikes → stored in episodic memory → consolidated into semantic after N exposures"""
        print("🐱 Demonstrating cat example...")
        
        results = []
        
        # Process "cat" multiple times to trigger consolidation
        for i in range(6):  # More than semantic_consolidation_threshold (5)
            print(f"   Processing 'cat' (exposure {i+1}/6)")
            
            result = self.process_input("cat", {"exposure": i+1, "example": "cat_demo"})
            results.append(result)
            
            # Small delay between exposures
            time.sleep(0.1)
        
        # Check if consolidation occurred
        cat_semantic = self.semantic_memory.find_semantic_memory("cat")
        
        demo_result = {
            "exposures": len(results),
            "consolidation_occurred": cat_semantic is not None,
            "semantic_memory": cat_semantic.dict() if cat_semantic else None,
            "processing_results": results
        }
        
        print(f"   ✅ Cat example complete:")
        print(f"      Exposures: {demo_result['exposures']}")
        print(f"      Consolidated: {demo_result['consolidation_occurred']}")
        
        return demo_result
    
    def get_memory_statistics(self) -> Dict[str, Any]:
        """Get detailed memory statistics"""
        # Working memory stats
        working_items = self.working_memory.get_all_items()
        working_expired = len(self.working_memory.items) - len(working_items)
        
        # Episodic memory stats
        episodic_traces = list(self.episodic_memory.traces.values())
        avg_access_count = np.mean([trace.access_count for trace in episodic_traces]) if episodic_traces else 0
        
        # Semantic memory stats
        semantic_memories = list(self.semantic_memory.semantic_memories.values())
        avg_consolidated_count = np.mean([mem.consolidated_count for mem in semantic_memories]) if semantic_memories else 0
        
        return {
            "working_memory": {
                "total_items": len(self.working_memory.items),
                "active_items": len(working_items),
                "expired_items": working_expired
            },
            "episodic_memory": {
                "total_traces": len(episodic_traces),
                "average_access_count": avg_access_count,
                "oldest_trace": min([trace.timestamp for trace in episodic_traces]) if episodic_traces else None,
                "newest_trace": max([trace.timestamp for trace in episodic_traces]) if episodic_traces else None
            },
            "semantic_memory": {
                "total_memories": len(semantic_memories),
                "average_consolidated_count": avg_consolidated_count,
                "concepts": [mem.concept for mem in semantic_memories]
            }
        }
