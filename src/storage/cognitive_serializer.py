#!/usr/bin/env python3
"""
Cognitive Architecture Serializer - Save/load complete cognitive architecture state
Implements snapshot v1 schema for CognitiveArchitecture serialization
"""

import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List
import numpy as np

from ..core.snapshot_schema import SnapshotV1, SnapshotMetadata, BackendState, CognitiveLayerState
from ..core.cognitive_architecture import CognitiveArchitecture
from ..core.cognitive_models import EpisodeTrace, WorkingMemoryItem, SemanticMemory


class CognitiveArchitectureSerializer:
    """
    Serializer for CognitiveArchitecture state.
    
    Implements snapshot v1 schema for complete brain state portability.
    """
    
    def __init__(self, storage_path: str = "./brain_snapshots"):
        """
        Initialize serializer.
        
        Args:
            storage_path: Directory path for storing snapshots
        """
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        print(f"💾 Cognitive Architecture Serializer initialized - storage: {self.storage_path}")
    
    def save(self, architecture: CognitiveArchitecture, snapshot_id: Optional[str] = None, description: Optional[str] = None) -> str:
        """
        Save complete cognitive architecture state to JSON snapshot.
        
        Args:
            architecture: CognitiveArchitecture instance to serialize
            snapshot_id: Optional snapshot ID (generated if not provided)
            description: Optional description for the snapshot
            
        Returns:
            Snapshot ID
        """
        if snapshot_id is None:
            snapshot_id = str(uuid.uuid4())
        
        print(f"💾 Saving cognitive architecture snapshot: {snapshot_id}")
        
        # 1. Backend state
        backend_state = BackendState(
            backend_name=architecture.backend.backend_name,
            state=architecture.backend.get_state(),
            config={
                "num_neurons": architecture.backend.num_neurons,
                "dt": architecture.backend.dt,
            }
        )
        
        # 2. Cognitive layer states
        cognitive_layers = {}
        
        # Sensory layer
        cognitive_layers["sensory"] = CognitiveLayerState(
            layer_name="sensory",
            state_data={
                "encoding_cache_size": len(architecture.sensory_layer.encoding_cache),
            }
        )
        
        # Associative layer
        cognitive_layers["associative"] = CognitiveLayerState(
            layer_name="associative",
            state_data={
                "association_matrix": architecture.associative_layer.association_matrix.tolist(),
                "co_activity_history_size": len(architecture.associative_layer.co_activity_history),
            }
        )
        
        # Working memory
        working_memory_items = {}
        for item_id, item in architecture.working_memory.items.items():
            working_memory_items[item_id] = {
                "item_id": item.item_id,
                "content": item.content,
                "timestamp": item.timestamp.isoformat(),
                "ttl": item.ttl,
                "access_count": item.access_count,
                "last_access": item.last_access.isoformat(),
            }
        
        cognitive_layers["working_memory"] = CognitiveLayerState(
            layer_name="working_memory",
            state_data={
                "items": working_memory_items,
                "max_items": architecture.config.max_working_memory_items,
                "ttl": architecture.config.working_memory_ttl,
            }
        )
        
        # Episodic memory
        episodic_traces = {}
        for trace_id, trace in architecture.episodic_memory.traces.items():
            episodic_traces[trace_id] = {
                "trace_id": trace.trace_id,
                "timestamp": trace.timestamp.isoformat(),
                "duration": trace.duration,
                "sensory_input": trace.sensory_input,
                "sensory_encoding": trace.sensory_encoding,
                "embedding": trace.embedding,
                "context": trace.context,
                "attention_weights": trace.attention_weights,
                "associations": list(trace.associations),
                "activation_strength": trace.activation_strength,
                "access_count": trace.access_count,
                "consolidation_level": trace.consolidation_level,
                "last_consolidation": trace.last_consolidation.isoformat() if trace.last_consolidation else None,
            }
        
        cognitive_layers["episodic_memory"] = CognitiveLayerState(
            layer_name="episodic_memory",
            state_data={
                "traces": episodic_traces,
                "max_traces": architecture.config.max_episodic_memories,
            }
        )
        
        # Semantic memory
        semantic_memories = {}
        for semantic_id, semantic in architecture.semantic_memory.semantic_memories.items():
            semantic_memories[semantic_id] = {
                "semantic_id": semantic.semantic_id,
                "concept": semantic.concept,
                "consolidated_traces": list(semantic.consolidated_traces),
                "consolidated_count": semantic.consolidated_count,
                "semantic_vector": semantic.semantic_vector,
                "activation_pattern": semantic.activation_pattern,
                "creation_time": semantic.creation_time.isoformat(),
                "last_update": semantic.last_update.isoformat(),
                "confidence": semantic.confidence,
            }
        
        cognitive_layers["semantic_memory"] = CognitiveLayerState(
            layer_name="semantic_memory",
            state_data={
                "semantic_memories": semantic_memories,
            }
        )
        
        # Executive layer
        cognitive_layers["executive"] = CognitiveLayerState(
            layer_name="executive",
            state_data={
                "arbitration_history_size": len(architecture.executive_layer.arbitration_history),
            }
        )
        
        # 3. System state
        system_state = {
            "config": architecture.config.dict(),
            "current_state": {
                "total_activations": architecture.current_state.total_activations,
                "consolidation_events": architecture.current_state.consolidation_events,
                "recall_events": architecture.current_state.recall_events,
            },
            "processing_history_size": len(architecture.processing_history),
        }
        
        # 4. Create snapshot
        snapshot = SnapshotV1(
            metadata=SnapshotMetadata(
                snapshot_id=snapshot_id,
                version="1.0",
                created_at=datetime.now(),
                backend_name=architecture.backend.backend_name,
                num_neurons=architecture.backend.num_neurons,
                description=description,
            ),
            backend_state=backend_state,
            cognitive_layers=cognitive_layers,
            system_state=system_state,
        )
        
        # 5. Save to file
        filename = f"{snapshot_id}.json"
        filepath = self.storage_path / filename
        
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(snapshot.dict(), f, indent=2, default=self._json_serializer)
        
        file_size = filepath.stat().st_size
        print(f"✅ Snapshot saved: {filepath} ({file_size:,} bytes)")
        print(f"   📊 Episodic traces: {len(episodic_traces)}")
        print(f"   🧠 Semantic memories: {len(semantic_memories)}")
        print(f"   💭 Working memory items: {len(working_memory_items)}")
        
        return snapshot_id
    
    def load(self, snapshot_id: str, architecture: Optional[CognitiveArchitecture] = None) -> CognitiveArchitecture:
        """
        Load cognitive architecture from snapshot.
        
        Args:
            snapshot_id: Snapshot ID to load
            architecture: Optional existing architecture to restore into (creates new if None)
            
        Returns:
            Restored CognitiveArchitecture instance
        """
        # Load snapshot file
        filename = f"{snapshot_id}.json"
        filepath = self.storage_path / filename
        
        if not filepath.exists():
            raise FileNotFoundError(f"Snapshot '{snapshot_id}' not found at {filepath}")
        
        print(f"📂 Loading snapshot: {snapshot_id}")
        
        with open(filepath, "r", encoding="utf-8") as f:
            snapshot_data = json.load(f)
        
        snapshot = SnapshotV1(**snapshot_data)
        
        # Create or use existing architecture
        if architecture is None:
            from ..core.cognitive_models import CognitiveConfig
            config = CognitiveConfig(**snapshot.system_state["config"])
            architecture = CognitiveArchitecture(config=config, backend_name=snapshot.metadata.backend_name)
        
        # 1. Restore backend state
        architecture.backend.set_state(snapshot.backend_state.state)
        
        # 2. Restore cognitive layers
        # Working memory
        working_memory_data = snapshot.cognitive_layers.get("working_memory", {}).state_data
        architecture.working_memory.items.clear()
        for item_id, item_data in working_memory_data.get("items", {}).items():
            item = WorkingMemoryItem(
                item_id=item_data["item_id"],
                content=item_data["content"],
                timestamp=datetime.fromisoformat(item_data["timestamp"]),
                ttl=item_data["ttl"],
                access_count=item_data["access_count"],
                last_access=datetime.fromisoformat(item_data["last_access"]),
            )
            architecture.working_memory.items[item_id] = item
        
        # Episodic memory
        episodic_data = snapshot.cognitive_layers.get("episodic_memory", {}).state_data
        architecture.episodic_memory.traces.clear()
        for trace_id, trace_data in episodic_data.get("traces", {}).items():
            trace = EpisodeTrace(
                trace_id=trace_data["trace_id"],
                timestamp=datetime.fromisoformat(trace_data["timestamp"]),
                duration=trace_data["duration"],
                sensory_input=trace_data["sensory_input"],
                sensory_encoding=trace_data["sensory_encoding"],
                embedding=trace_data.get("embedding"),
                context=trace_data.get("context", {}),
                attention_weights=trace_data.get("attention_weights"),
                associations=set(trace_data.get("associations", [])),
                activation_strength=trace_data.get("activation_strength", 1.0),
                access_count=trace_data.get("access_count", 1),
                consolidation_level=trace_data.get("consolidation_level", 0.0),
                last_consolidation=datetime.fromisoformat(trace_data["last_consolidation"]) if trace_data.get("last_consolidation") else None,
            )
            architecture.episodic_memory.traces[trace_id] = trace
        
        # Semantic memory
        semantic_data = snapshot.cognitive_layers.get("semantic_memory", {}).state_data
        architecture.semantic_memory.semantic_memories.clear()
        for semantic_id, semantic_data_item in semantic_data.get("semantic_memories", {}).items():
            semantic = SemanticMemory(
                semantic_id=semantic_data_item["semantic_id"],
                concept=semantic_data_item["concept"],
                consolidated_traces=set(semantic_data_item["consolidated_traces"]),
                consolidated_count=semantic_data_item["consolidated_count"],
                semantic_vector=semantic_data_item["semantic_vector"],
                activation_pattern=semantic_data_item["activation_pattern"],
                creation_time=datetime.fromisoformat(semantic_data_item["creation_time"]),
                last_update=datetime.fromisoformat(semantic_data_item["last_update"]),
                confidence=semantic_data_item.get("confidence", 0.5),
            )
            architecture.semantic_memory.semantic_memories[semantic_id] = semantic
        
        # Associative layer
        associative_data = snapshot.cognitive_layers.get("associative", {}).state_data
        if "association_matrix" in associative_data:
            architecture.associative_layer.association_matrix = np.array(associative_data["association_matrix"])
        
        # 3. Restore system state
        architecture.current_state.total_activations = snapshot.system_state["current_state"]["total_activations"]
        architecture.current_state.consolidation_events = snapshot.system_state["current_state"]["consolidation_events"]
        architecture.current_state.recall_events = snapshot.system_state["current_state"]["recall_events"]
        
        print(f"✅ Snapshot loaded successfully")
        print(f"   📊 Episodic traces: {len(architecture.episodic_memory.traces)}")
        print(f"   🧠 Semantic memories: {len(architecture.semantic_memory.semantic_memories)}")
        print(f"   💭 Working memory items: {len(architecture.working_memory.items)}")
        
        return architecture
    
    def list_snapshots(self) -> List[Dict[str, Any]]:
        """List all available snapshots"""
        snapshots = []
        
        for json_file in self.storage_path.glob("*.json"):
            try:
                with open(json_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    metadata = data.get("metadata", {})
                    
                    snapshots.append({
                        "snapshot_id": metadata.get("snapshot_id", json_file.stem),
                        "created_at": metadata.get("created_at", "Unknown"),
                        "version": metadata.get("version", "Unknown"),
                        "backend_name": metadata.get("backend_name", "Unknown"),
                        "num_neurons": metadata.get("num_neurons", 0),
                        "description": metadata.get("description"),
                        "file_size": json_file.stat().st_size,
                    })
            except (json.JSONDecodeError, KeyError) as e:
                print(f"⚠️ Skipping invalid snapshot {json_file.name}: {e}")
                continue
        
        # Sort by creation date (newest first)
        snapshots.sort(key=lambda x: x["created_at"], reverse=True)
        
        return snapshots
    
    @staticmethod
    def _json_serializer(obj):
        """Custom JSON serializer for complex objects"""
        if hasattr(obj, "isoformat"):
            return obj.isoformat()
        elif hasattr(obj, "__dict__"):
            return obj.__dict__
        else:
            return str(obj)

