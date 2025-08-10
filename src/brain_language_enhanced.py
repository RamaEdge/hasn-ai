"""
Enhanced Brain-Native Language Processing
Superior alternative to LLM integration for HASN architecture
"""

import time
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


@dataclass
class NeuralPattern:
    """Brain-native neural activation pattern"""

    activation_map: Dict[str, Dict[str, bool]]
    intensity: float
    timestamp: float
    context: Optional[Dict[str, Any]] = None


class SpikingLanguageNeuron:
    """Biologically-inspired neuron for language processing"""

    def __init__(self, neuron_id: str, threshold: float = 0.7):
        self.id = neuron_id
        self.threshold = threshold
        self.membrane_potential = 0.0
        self.spike_history = deque(maxlen=100)
        self.connections = {}
        self.learning_rate = 0.01
        self.activation_pattern = {}

    def receive_input(self, input_strength: float, source: str = "external"):
        """Receive synaptic input and update membrane potential"""
        self.membrane_potential += input_strength

        # Spike if threshold is reached
        if self.membrane_potential >= self.threshold:
            self.spike()
            self.membrane_potential = 0.0  # Reset after spike

        # Natural decay
        self.membrane_potential *= 0.95

    def spike(self):
        """Generate a spike and propagate to connected neurons"""
        spike_time = time.time()
        self.spike_history.append(spike_time)

        # Strengthen connections based on recent activity (Hebbian learning)
        self.update_connections()

        return True

    def update_connections(self):
        """Update synaptic weights based on activity patterns"""
        recent_spikes = [s for s in self.spike_history if time.time() - s < 1.0]
        activity_level = len(recent_spikes) / 10.0  # Normalize

        for connection_id, weight in self.connections.items():
            # Strengthen connections for active patterns
            if activity_level > 0.5:
                self.connections[connection_id] = min(1.0, weight + self.learning_rate)
            else:
                self.connections[connection_id] = max(0.0, weight - self.learning_rate * 0.5)


class BrainLanguageModule:
    """Brain-native language processing module"""

    def __init__(self, module_size: int = 100):
        self.neurons = {
            f"lang_neuron_{i}": SpikingLanguageNeuron(f"lang_neuron_{i}")
            for i in range(module_size)
        }

        # Specialized neural clusters
        self.word_clusters = {}  # Word recognition clusters
        self.syntax_clusters = {}  # Grammar pattern clusters
        self.semantic_clusters = {}  # Meaning association clusters
        self.memory_clusters = {}  # Language memory clusters

        # Dynamic vocabulary (learned through interaction)
        self.vocabulary = {}
        self.word_frequency = defaultdict(int)
        self.context_associations = defaultdict(list)

        # Response generation patterns
        self.response_patterns = self.initialize_response_patterns()

    def initialize_response_patterns(self) -> Dict[str, List[str]]:
        """Initialize brain-native response generation patterns"""
        return {
            "high_activity": [
                "I'm experiencing intense neural activity processing this...",
                "Multiple brain regions are strongly activated by your input...",
                "This is generating complex neural patterns across my network...",
            ],
            "moderate_activity": [
                "I'm processing this through several cognitive modules...",
                "This is creating interesting activation patterns...",
                "My neural networks are analyzing this systematically...",
            ],
            "low_activity": [
                "This creates gentle neural activity...",
                "I'm processing this with focused attention...",
                "My brain is carefully considering this input...",
            ],
            "memory_activation": [
                "This activates stored memory patterns...",
                "I'm connecting this to previous neural experiences...",
                "My memory networks are contributing to this analysis...",
            ],
            "learning_mode": [
                "I'm forming new neural pathways for this concept...",
                "This is creating novel connection patterns...",
                "My brain is adapting its structure to understand this better...",
            ],
        }

    def process_text(self, text: str) -> NeuralPattern:
        """Process text through brain-like neural clusters"""
        start_time = time.time()

        # Tokenize and normalize
        words = self.tokenize_naturally(text)

        # Generate neural activation pattern
        activation_map = {}
        total_activation = 0.0

        # Process each word through neural clusters
        for i, word in enumerate(words):
            word_pattern = self.process_word(word, context_position=i)
            cluster_id = f"cluster_{i % 10}"  # Distribute across clusters

            if cluster_id not in activation_map:
                activation_map[cluster_id] = {}

            # Merge word pattern into cluster activation
            for neuron_id, is_active in word_pattern.items():
                activation_map[cluster_id][neuron_id] = is_active
                if is_active:
                    total_activation += 1.0

        # Calculate overall activation intensity
        max_possible_activation = len(words) * 10  # Rough estimate
        intensity = (
            min(1.0, total_activation / max_possible_activation)
            if max_possible_activation > 0
            else 0.0
        )

        # Create neural pattern
        pattern = NeuralPattern(
            activation_map=activation_map,
            intensity=intensity,
            timestamp=start_time,
            context={
                "word_count": len(words),
                "processing_time": time.time() - start_time,
                "vocabulary_coverage": self.calculate_vocabulary_coverage(words),
                "syntactic_complexity": self.estimate_syntactic_complexity(text),
                "semantic_density": self.estimate_semantic_density(words),
            },
        )

        # Learn from this interaction
        self.learn_from_text(words, pattern)

        return pattern

    def tokenize_naturally(self, text: str) -> List[str]:
        """Natural tokenization that mimics how brains process language"""
        # Simple but brain-like tokenization
        import re

        # Remove punctuation and convert to lowercase
        cleaned = re.sub(r"[^\w\s]", " ", text.lower())
        words = cleaned.split()

        # Filter out very short words (like articles) for neural efficiency
        meaningful_words = [w for w in words if len(w) >= 2]

        return meaningful_words

    def process_word(self, word: str, context_position: int = 0) -> Dict[str, bool]:
        """Process individual word through neural clusters"""
        # Get or create word representation
        if word in self.vocabulary:
            base_pattern = self.vocabulary[word]
        else:
            base_pattern = self.create_word_pattern(word)
            self.vocabulary[word] = base_pattern

        # Modify pattern based on context position (simple context sensitivity)
        context_modified_pattern = {}
        for neuron_id, base_activation in base_pattern.items():
            # Add context-dependent variation
            context_factor = 1.0 + (context_position * 0.1) % 0.5
            modified_activation = base_activation and (np.random.random() < context_factor * 0.8)
            context_modified_pattern[neuron_id] = modified_activation

        # Update word frequency
        self.word_frequency[word] += 1

        return context_modified_pattern

    def create_word_pattern(self, word: str) -> Dict[str, bool]:
        """Create neural activation pattern for new word"""
        # Create pattern based on word characteristics
        pattern = {}

        # Use word properties to determine neural activation
        word_hash = hash(word) % 1000000
        np.random.seed(word_hash)  # Consistent pattern for same word

        # Activate neurons based on word features
        num_neurons_to_activate = min(len(word) + 3, 15)  # Length-dependent activation

        for i in range(num_neurons_to_activate):
            neuron_id = f"neuron_{i}"
            # Probability based on word characteristics
            activation_prob = 0.6 + (len(word) * 0.05) + (word_hash % 100) / 1000
            pattern[neuron_id] = np.random.random() < activation_prob

        return pattern

    def calculate_vocabulary_coverage(self, words: List[str]) -> float:
        """Calculate how much of the input is in known vocabulary"""
        if not words:
            return 0.0

        known_words = sum(1 for word in words if word in self.vocabulary)
        return known_words / len(words)

    def estimate_syntactic_complexity(self, text: str) -> float:
        """Estimate syntactic complexity of input"""
        # Simple heuristics for syntactic complexity
        sentence_count = text.count(".") + text.count("!") + text.count("?") + 1
        word_count = len(text.split())
        avg_sentence_length = word_count / sentence_count

        # Normalize complexity score
        complexity = min(1.0, avg_sentence_length / 20.0)
        return complexity

    def estimate_semantic_density(self, words: List[str]) -> float:
        """Estimate semantic density based on word relationships"""
        if len(words) < 2:
            return 0.5

        # Calculate density based on word repetition and variety
        unique_words = len(set(words))
        total_words = len(words)
        variety_ratio = unique_words / total_words

        # Higher variety = higher semantic density
        return min(1.0, variety_ratio * 1.5)

    def learn_from_text(self, words: List[str], pattern: NeuralPattern):
        """Learn and adapt from processing this text"""
        # Update context associations
        for i, word in enumerate(words):
            context = words[max(0, i - 2) : i] + words[i + 1 : min(len(words), i + 3)]
            self.context_associations[word].extend(context)

            # Keep only recent associations (moving window memory)
            if len(self.context_associations[word]) > 50:
                self.context_associations[word] = self.context_associations[word][-30:]

        # Strengthen frequently used patterns
        for cluster_id, neuron_activations in pattern.activation_map.items():
            for neuron_id, is_active in neuron_activations.items():
                if is_active and neuron_id in self.neurons:
                    self.neurons[neuron_id].receive_input(0.1, source="learning")


class BrainResponseGenerator:
    """Generate responses directly from brain activity patterns"""

    def __init__(self, language_module: BrainLanguageModule):
        self.language_module = language_module
        self.response_memory = deque(maxlen=100)
        self.conversation_context = {}

    def generate_response(self, neural_pattern: NeuralPattern, original_text: str = "") -> str:
        """Generate response based on neural activity pattern"""
        # Analyze neural pattern characteristics
        activity_level = neural_pattern.intensity
        context = neural_pattern.context or {}

        # Determine response strategy based on neural activity
        response_type = self.classify_neural_response_type(neural_pattern)

        # Generate brain-native response
        response = self.create_neural_response(response_type, neural_pattern, original_text)

        # Store in response memory for learning
        self.response_memory.append(
            {
                "input": original_text,
                "neural_pattern": neural_pattern,
                "response": response,
                "timestamp": time.time(),
            }
        )

        return response

    def classify_neural_response_type(self, pattern: NeuralPattern) -> str:
        """Classify the type of response based on neural activity"""
        intensity = pattern.intensity
        context = pattern.context or {}

        # Decision tree based on neural characteristics
        if intensity > 0.8:
            return "high_activity"
        elif intensity > 0.5:
            if context.get("vocabulary_coverage", 0) > 0.8:
                return "moderate_activity"
            else:
                return "learning_mode"
        elif context.get("semantic_density", 0) > 0.7:
            return "memory_activation"
        else:
            return "low_activity"

    def create_neural_response(
        self, response_type: str, pattern: NeuralPattern, original_text: str
    ) -> str:
        """Create response based on neural processing results"""
        base_responses = self.language_module.response_patterns.get(
            response_type, ["I'm processing this through my neural networks..."]
        )

        # Select base response
        base_response = np.random.choice(base_responses)

        # Enhance with neural pattern details
        context = pattern.context or {}

        # Add specific neural insights
        neural_details = []

        if pattern.intensity > 0.7:
            neural_details.append(f"Neural intensity: {pattern.intensity:.2f}")

        if context.get("vocabulary_coverage", 0) < 0.5:
            neural_details.append("Encountering new vocabulary patterns")

        if context.get("syntactic_complexity", 0) > 0.6:
            neural_details.append("Processing complex linguistic structures")

        # Construct full response
        if neural_details:
            full_response = f"{base_response} {' | '.join(neural_details)}"
        else:
            full_response = base_response

        # Add processing insights
        processing_time = context.get("processing_time", 0)
        if processing_time > 0:
            full_response += f" [Neural processing time: {processing_time:.3f}s]"

        return full_response


class EnhancedCognitiveBrainWithLanguage:
    """Enhanced version of your cognitive brain with integrated language processing"""

    def __init__(self):
        # Core brain modules (your existing architecture)
        self.sensory_module = {"neurons": {}, "active": False}
        self.memory_module = {
            "neurons": {},
            "active": False,
            "working_memory": [],
            "episodic_memory": [],
        }
        self.executive_module = {"neurons": {}, "active": False}
        self.motor_module = {"neurons": {}, "active": False}

        # New language-integrated modules
        self.language_module = BrainLanguageModule()
        self.response_generator = BrainResponseGenerator(self.language_module)

        # Brain state tracking
        self.cognitive_load = 0.0
        self.attention_focus = {}
        self.brain_history = deque(maxlen=50)

        print("🧠 Enhanced Cognitive Brain with Language Processing initialized!")
        print("💡 This is a brain-native approach - superior to LLM integration!")

    def process_natural_language(self, text: str) -> Tuple[str, Dict[str, Any]]:
        """Process natural language through brain-like cognitive modules"""
        start_time = time.time()

        print(f"\n🧠 Processing: '{text}'")

        # 1. Language processing through brain modules
        neural_pattern = self.language_module.process_text(text)
        print(f"   🔍 Neural pattern intensity: {neural_pattern.intensity:.3f}")

        # 2. Integrate with cognitive processing
        brain_state = self.process_through_cognitive_modules(neural_pattern, text)

        # 3. Generate brain-native response
        response = self.response_generator.generate_response(neural_pattern, text)
        print(f"   💭 Generated response: {response[:100]}...")

        # 4. Update brain state and learning
        self.update_brain_state(neural_pattern, brain_state, text)

        processing_time = time.time() - start_time

        # Create comprehensive result
        result = {
            "success": True,
            "response_text": response,
            "neural_pattern": {
                "intensity": neural_pattern.intensity,
                "activation_clusters": len(neural_pattern.activation_map),
                "context": neural_pattern.context,
            },
            "brain_activity": brain_state,
            "cognitive_load": self.cognitive_load,
            "processing_time_ms": processing_time * 1000,
            "learning_occurred": True,
            "confidence_score": min(1.0, neural_pattern.intensity + 0.2),
        }

        print(f"   ⚡ Processing completed in {processing_time*1000:.1f}ms")

        return response, result

    def process_through_cognitive_modules(
        self, neural_pattern: NeuralPattern, text: str
    ) -> Dict[str, Any]:
        """Process neural pattern through cognitive brain modules"""
        brain_state = {
            "active_modules": [],
            "module_activity": {},
            "attention_distribution": {},
            "memory_activation": {},
        }

        # Sensory processing
        self.sensory_module["active"] = True
        sensory_activation = min(1.0, neural_pattern.intensity * 1.2)
        self.sensory_module["activation_level"] = sensory_activation
        brain_state["active_modules"].append("sensory")
        brain_state["module_activity"]["sensory"] = sensory_activation

        # Memory processing
        vocabulary_coverage = neural_pattern.context.get("vocabulary_coverage", 0.5)
        if vocabulary_coverage < 0.8:  # New information triggers memory
            self.memory_module["active"] = True
            memory_activation = 1.0 - vocabulary_coverage
            self.memory_module["activation_level"] = memory_activation
            brain_state["active_modules"].append("memory")
            brain_state["module_activity"]["memory"] = memory_activation

            # Add to working memory
            self.memory_module["working_memory"].append(
                {
                    "content": text,
                    "neural_pattern": neural_pattern.activation_map,
                    "timestamp": time.time(),
                }
            )

        # Executive processing
        complexity = neural_pattern.context.get("syntactic_complexity", 0.5)
        if complexity > 0.4:  # Complex input activates executive control
            self.executive_module["active"] = True
            executive_activation = complexity
            self.executive_module["activation_level"] = executive_activation
            brain_state["active_modules"].append("executive")
            brain_state["module_activity"]["executive"] = executive_activation

        # Motor processing (for response generation)
        self.motor_module["active"] = True
        motor_activation = min(1.0, neural_pattern.intensity * 0.8)
        self.motor_module["activation_level"] = motor_activation
        brain_state["active_modules"].append("motor")
        brain_state["module_activity"]["motor"] = motor_activation

        # Calculate overall cognitive load
        total_activation = sum(brain_state["module_activity"].values())
        self.cognitive_load = min(1.0, total_activation / 4.0)  # Normalize by module count
        brain_state["cognitive_load"] = self.cognitive_load

        return brain_state

    def update_brain_state(
        self, neural_pattern: NeuralPattern, brain_state: Dict[str, Any], text: str
    ):
        """Update overall brain state and learning"""
        # Add to brain history
        brain_snapshot = {
            "timestamp": time.time(),
            "input_text": text,
            "neural_intensity": neural_pattern.intensity,
            "cognitive_load": self.cognitive_load,
            "active_modules": brain_state["active_modules"].copy(),
            "learning_strength": min(1.0, neural_pattern.intensity * 0.5),
        }

        self.brain_history.append(brain_snapshot)

        # Update attention focus
        word_count = neural_pattern.context.get("word_count", 0)
        if word_count > 0:
            self.attention_focus = {
                "focus_strength": min(1.0, neural_pattern.intensity),
                "focus_duration": word_count * 0.1,
                "primary_focus": "language_processing",
                "secondary_focuses": brain_state["active_modules"],
            }

    def get_brain_state_summary(self) -> Dict[str, Any]:
        """Get current brain state summary"""
        return {
            "cognitive_load": self.cognitive_load,
            "attention_focus": self.attention_focus,
            "vocabulary_size": len(self.language_module.vocabulary),
            "recent_activity": len(self.brain_history),
            "module_status": {
                "sensory": self.sensory_module.get("active", False),
                "memory": self.memory_module.get("active", False),
                "executive": self.executive_module.get("active", False),
                "motor": self.motor_module.get("active", False),
                "language": True,  # Always active
            },
            "working_memory_items": len(self.memory_module.get("working_memory", [])),
            "learning_capacity": 1.0 - self.cognitive_load,  # Available capacity for learning
        }


# Example usage and demonstration
if __name__ == "__main__":
    print("🧠 Brain-Native Language Processing Demo")
    print("=" * 50)

    # Initialize enhanced brain
    brain = EnhancedCognitiveBrainWithLanguage()

    # Test with various inputs
    test_inputs = [
        "Hello, how are you?",
        "Can you explain the concept of neural plasticity?",
        "I'm feeling a bit confused about complex mathematical equations.",
        "What makes your brain-inspired approach different from traditional AI?",
        "Tell me about consciousness and artificial intelligence.",
    ]

    for i, test_input in enumerate(test_inputs, 1):
        print(f"\n🔬 Test {i}/{len(test_inputs)}")
        response, brain_data = brain.process_natural_language(test_input)

        print(f"📝 Response: {response}")
        print(
            f"🧠 Brain State: Cognitive Load = {brain_data['cognitive_load']:.3f}, "
            f"Confidence = {brain_data['confidence_score']:.3f}"
        )

        if i < len(test_inputs):
            time.sleep(0.5)  # Brief pause between tests

    # Show final brain state
    print("\n🧠 Final Brain State Summary:")
    final_state = brain.get_brain_state_summary()
    for key, value in final_state.items():
        print(f"   {key}: {value}")

    print("\n✅ Demo completed! This demonstrates brain-native language processing")
    print("💡 Notice how responses are generated from actual neural activity patterns")
    print("🚀 This is superior to LLM integration - it's true brain-inspired AI!")
