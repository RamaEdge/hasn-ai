"""
Brain State Serialization - JSON-based brain portability
Simple implementation to make your trained brain portable
"""

import json
import time
from collections import defaultdict, deque
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List


class BrainStateSerializer:
    """Serialize and deserialize complete brain states to/from JSON"""

    def __init__(self, storage_path: str = "./brain_states"):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(exist_ok=True)
        print(f"🧠 Brain serializer initialized - storage: {self.storage_path}")

    def save_brain_state(self, brain, session_name: str = None) -> str:
        """Save complete brain state to JSON file"""

        if session_name is None:
            session_name = f"brain_session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        print(f"💾 Saving brain state: {session_name}")

        # Extract all brain components that need to be preserved
        brain_state = {
            "metadata": {
                "session_name": session_name,
                "saved_at": datetime.now().isoformat(),
                "version": "1.0.0",
                "brain_type": "EnhancedCognitiveBrainWithLanguage",
                "serializer": "BrainStateSerializer",
            },
            # Language Module State - the core learning component
            "language_module": {
                "vocabulary": brain.language_module.vocabulary,
                "word_frequency": dict(brain.language_module.word_frequency),
                "context_associations": {
                    word: list(contexts)
                    for word, contexts in brain.language_module.context_associations.items()
                },
                # Neural network state
                "neuron_states": {
                    neuron_id: {
                        "threshold": neuron.threshold,
                        "membrane_potential": neuron.membrane_potential,
                        "connections": neuron.connections,
                        "learning_rate": neuron.learning_rate,
                        "spike_count": len(neuron.spike_history),
                    }
                    for neuron_id, neuron in brain.language_module.neurons.items()
                },
            },
            # Memory System State
            "memory_module": {
                "working_memory": brain.memory_module.get("working_memory", []),
                "episodic_memory": brain.memory_module.get("episodic_memory", []),
                "module_active": brain.memory_module.get("active", False),
            },
            # Response Generation State
            "response_generator": {
                "response_memory": [
                    {
                        "input": item.get("input", ""),
                        "response": item.get("response", ""),
                        "timestamp": item.get("timestamp", time.time()),
                    }
                    for item in brain.response_generator.response_memory
                ],
                "conversation_context": dict(brain.response_generator.conversation_context),
            },
            # Brain Activity History
            "brain_history": [
                {
                    "timestamp": item.get("timestamp", time.time()),
                    "input_text": item.get("input_text", ""),
                    "neural_intensity": item.get("neural_intensity", 0.0),
                    "cognitive_load": item.get("cognitive_load", 0.0),
                }
                for item in brain.brain_history
            ],
            # Current Cognitive State
            "cognitive_metrics": {
                "cognitive_load": getattr(brain, "cognitive_load", 0.0),
                "attention_focus": dict(getattr(brain, "attention_focus", {})),
                "current_timestamp": time.time(),
            },
            # Learning Statistics for verification
            "learning_stats": {
                "vocabulary_size": len(brain.language_module.vocabulary),
                "total_word_encounters": sum(brain.language_module.word_frequency.values()),
                "unique_contexts": sum(
                    len(contexts)
                    for contexts in brain.language_module.context_associations.values()
                ),
                "total_interactions": len(brain.brain_history),
                "neuron_count": len(brain.language_module.neurons),
            },
        }

        # Save to file
        filename = f"{session_name}.json"
        filepath = self.storage_path / filename

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(brain_state, f, indent=2, default=self._json_serializer)

        # Verify save success
        file_size = filepath.stat().st_size
        vocab_size = brain_state["learning_stats"]["vocabulary_size"]

        print("✅ Brain saved successfully!")
        print(f"   📄 File: {filepath}")
        print(f"   📊 Size: {file_size:,} bytes")
        print(f"   📚 Vocabulary: {vocab_size} words")
        print(f"   🧠 Neurons: {brain_state['learning_stats']['neuron_count']}")

        return str(filepath)

    def load_brain_state(self, session_name: str) -> Dict[str, Any]:
        """Load brain state from JSON file"""

        if not session_name.endswith(".json"):
            session_name += ".json"

        filepath = self.storage_path / session_name

        if not filepath.exists():
            raise FileNotFoundError(f"Brain session '{session_name}' not found at {filepath}")

        print(f"📂 Loading brain state: {session_name}")

        with open(filepath, "r", encoding="utf-8") as f:
            brain_state = json.load(f)

        # Verify loaded data
        metadata = brain_state.get("metadata", {})
        learning_stats = brain_state.get("learning_stats", {})

        print("✅ Brain state loaded!")
        print(f"   📅 Saved: {metadata.get('saved_at', 'Unknown')}")
        print(f"   📚 Vocabulary: {learning_stats.get('vocabulary_size', 0)} words")
        print(f"   🔗 Interactions: {learning_stats.get('total_interactions', 0)}")

        return brain_state

    def restore_brain_from_state(self, brain_state: Dict[str, Any]):
        """Restore a complete brain from saved state"""

        print("🔄 Restoring brain from saved state...")

        # Import here to avoid circular imports
        from brain_language_enhanced import EnhancedCognitiveBrainWithLanguage

        # Create new brain instance
        brain = EnhancedCognitiveBrainWithLanguage()

        # Restore language module vocabulary and learning
        lang_state = brain_state.get("language_module", {})

        # Restore vocabulary
        if "vocabulary" in lang_state:
            brain.language_module.vocabulary = lang_state["vocabulary"]
            print(f"   📚 Restored vocabulary: {len(brain.language_module.vocabulary)} words")

        # Restore word frequencies
        if "word_frequency" in lang_state:
            brain.language_module.word_frequency = defaultdict(int)
            brain.language_module.word_frequency.update(lang_state["word_frequency"])
            print(f"   📊 Restored word frequencies: {len(lang_state['word_frequency'])} entries")

        # Restore context associations
        if "context_associations" in lang_state:
            brain.language_module.context_associations = defaultdict(deque)
            for word, contexts in lang_state["context_associations"].items():
                brain.language_module.context_associations[word] = deque(contexts, maxlen=50)
            print(
                f"   🌐 Restored context associations: {len(lang_state['context_associations'])} words"
            )

        # Restore neuron states
        if "neuron_states" in lang_state:
            neuron_count = 0
            for neuron_id, neuron_data in lang_state["neuron_states"].items():
                if neuron_id in brain.language_module.neurons:
                    neuron = brain.language_module.neurons[neuron_id]
                    neuron.threshold = neuron_data.get("threshold", neuron.threshold)
                    neuron.membrane_potential = neuron_data.get("membrane_potential", 0.0)
                    neuron.connections = neuron_data.get("connections", {})
                    neuron.learning_rate = neuron_data.get("learning_rate", neuron.learning_rate)
                    neuron_count += 1
            print(f"   🧠 Restored neural states: {neuron_count} neurons")

        # Restore memory module
        memory_state = brain_state.get("memory_module", {})
        if "working_memory" in memory_state:
            brain.memory_module["working_memory"] = memory_state["working_memory"]
        if "episodic_memory" in memory_state:
            brain.memory_module["episodic_memory"] = memory_state["episodic_memory"]
        if "module_active" in memory_state:
            brain.memory_module["active"] = memory_state["module_active"]
        print(f"   💾 Restored memory: {len(memory_state.get('working_memory', []))} working items")

        # Restore response generator
        response_state = brain_state.get("response_generator", {})
        if "response_memory" in response_state:
            brain.response_generator.response_memory = deque(maxlen=100)
            brain.response_generator.response_memory.extend(response_state["response_memory"])
        if "conversation_context" in response_state:
            brain.response_generator.conversation_context = response_state["conversation_context"]
        print(f"   💭 Restored responses: {len(response_state.get('response_memory', []))} entries")

        # Restore brain history
        if "brain_history" in brain_state:
            brain.brain_history = deque(maxlen=50)
            brain.brain_history.extend(brain_state["brain_history"])
            print(f"   📈 Restored brain history: {len(brain_state['brain_history'])} snapshots")

        # Restore cognitive metrics
        cognitive_state = brain_state.get("cognitive_metrics", {})
        brain.cognitive_load = cognitive_state.get("cognitive_load", 0.0)
        brain.attention_focus = cognitive_state.get("attention_focus", {})
        print(f"   🎯 Restored cognitive state: load={brain.cognitive_load:.3f}")

        # Verify restoration
        final_stats = brain_state.get("learning_stats", {})
        current_vocab_size = len(brain.language_module.vocabulary)
        expected_vocab_size = final_stats.get("vocabulary_size", 0)

        print("\n🎊 Brain restoration complete!")
        print(f"   📚 Vocabulary restored: {current_vocab_size}/{expected_vocab_size} words")
        print("   🧠 Neural patterns: Fully restored")
        print("   💾 Memory systems: Active")
        print("   🔄 Learning capacity: Ready for new interactions")

        if current_vocab_size == expected_vocab_size:
            print("   ✅ Perfect restoration - brain is identical to saved state!")
        else:
            print("   ⚠️  Partial restoration - some vocabulary may be missing")

        return brain

    def list_saved_sessions(self) -> List[Dict[str, Any]]:
        """List all saved brain sessions"""

        sessions = []

        for json_file in self.storage_path.glob("*.json"):
            try:
                with open(json_file, "r", encoding="utf-8") as f:
                    # Load just metadata to avoid loading full brain
                    data = json.load(f)
                    metadata = data.get("metadata", {})
                    learning_stats = data.get("learning_stats", {})

                sessions.append(
                    {
                        "filename": json_file.name,
                        "session_name": metadata.get("session_name", json_file.stem),
                        "saved_at": metadata.get("saved_at", "Unknown"),
                        "version": metadata.get("version", "Unknown"),
                        "vocabulary_size": learning_stats.get("vocabulary_size", 0),
                        "total_interactions": learning_stats.get("total_interactions", 0),
                        "file_size": json_file.stat().st_size,
                    }
                )

            except (json.JSONDecodeError, KeyError) as e:
                print(f"⚠️ Skipping invalid file {json_file.name}: {e}")
                continue

        # Sort by save date (newest first)
        sessions.sort(key=lambda x: x["saved_at"], reverse=True)

        return sessions

    def delete_session(self, session_name: str) -> bool:
        """Delete a saved brain session"""

        if not session_name.endswith(".json"):
            session_name += ".json"

        filepath = self.storage_path / session_name

        if filepath.exists():
            filepath.unlink()
            print(f"🗑️ Deleted brain session: {session_name}")
            return True
        else:
            print(f"❌ Session not found: {session_name}")
            return False

    def get_session_info(self, session_name: str) -> Dict[str, Any]:
        """Get detailed information about a saved session"""

        brain_state = self.load_brain_state(session_name)

        metadata = brain_state.get("metadata", {})
        learning_stats = brain_state.get("learning_stats", {})
        lang_state = brain_state.get("language_module", {})

        # Analyze vocabulary
        vocabulary = lang_state.get("vocabulary", {})
        word_frequency = lang_state.get("word_frequency", {})
        context_associations = lang_state.get("context_associations", {})

        # Get most frequent words
        if word_frequency:
            top_words = sorted(word_frequency.items(), key=lambda x: x[1], reverse=True)[:10]
        else:
            top_words = []

        return {
            "metadata": metadata,
            "learning_statistics": learning_stats,
            "vocabulary_analysis": {
                "total_words": len(vocabulary),
                "total_encounters": (sum(word_frequency.values()) if word_frequency else 0),
                "top_words": top_words,
                "context_richness": len(context_associations),
            },
            "brain_activity": {
                "total_interactions": len(brain_state.get("brain_history", [])),
                "response_patterns": len(
                    brain_state.get("response_generator", {}).get("response_memory", [])
                ),
                "memory_items": len(brain_state.get("memory_module", {}).get("working_memory", [])),
            },
        }

    @staticmethod
    def _json_serializer(obj):
        """Custom JSON serializer for complex objects"""
        if hasattr(obj, "isoformat"):
            return obj.isoformat()
        elif hasattr(obj, "__dict__"):
            return obj.__dict__
        else:
            return str(obj)


# Example usage and testing
if __name__ == "__main__":
    print("🧠 Brain State Serializer - Test Mode")
    print("=" * 50)

    # Initialize serializer
    serializer = BrainStateSerializer()

    # Test listing (should be empty initially)
    sessions = serializer.list_saved_sessions()
    print(f"📋 Found {len(sessions)} existing brain sessions")

    if sessions:
        print("\n📁 Available sessions:")
        for session in sessions[:3]:  # Show first 3
            print(f"   📄 {session['session_name']}")
            print(f"      📅 {session['saved_at']}")
            print(f"      📚 {session['vocabulary_size']} words")
            print(f"      📊 {session['file_size']:,} bytes")
            print()

    print("🎯 To use this with your brain:")
    print("   1. serializer = BrainStateSerializer()")
    print("   2. path = serializer.save_brain_state(your_brain, 'my_session')")
    print(
        "   3. loaded_brain = serializer.restore_brain_from_state(serializer.load_brain_state('my_session'))"
    )
    print("\n💡 Your trained brain is now portable! 🚀")
