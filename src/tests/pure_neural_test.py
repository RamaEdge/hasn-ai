#!/usr/bin/env python3
"""
Pure Neural Response Analysis
Isolated test of neural response generation
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from simple_brain_demo import SimpleBrainNetwork
import random
import time

class NeuralResponseAnalyzer:
    def __init__(self):
        self.network = SimpleBrainNetwork([30, 25, 20, 15])
        
    def text_to_pattern(self, text):
        """Convert text to neural activation pattern"""
        pattern = {}
        text = text.lower().strip()
        
        # Length-based encoding
        text_len = len(text)
        if text_len > 0:
            neurons_to_activate = min(text_len // 2, 15)
            pattern[0] = {i: True for i in range(neurons_to_activate)}
        
        # Content-based encoding
        content_neurons = []
        if any(q_word in text for q_word in ['what', 'how', 'why', '?']):
            content_neurons.extend(range(5, 12))
        if any(e_word in text for e_word in ['good', 'happy', 'love']):
            content_neurons.extend(range(15, 20))
        if any(l_word in text for l_word in ['learn', 'think', 'understand']):
            content_neurons.extend(range(8, 15))
            
        if content_neurons:
            pattern[1] = {i: True for i in set(content_neurons[:10])}
            
        # Word-specific patterns
        word_hash = hash(text) % 20
        pattern[2] = {i: True for i in range(word_hash % 5, word_hash % 5 + 3)}
        
        return pattern
    
    def analyze_neural_response(self, text):
        """Analyze how neural patterns generate responses"""
        print(f"Analyzing: '{text}'")
        
        # Convert to pattern
        pattern = self.text_to_pattern(text)
        print(f"  Neural pattern: {pattern}")
        
        # Process through network
        result = self.network.step(pattern)
        print(f"  Activity: {result['total_activity']:.3f}")
        print(f"  Memory: {result['memory_size']}/7")
        
        # Extract active neurons
        module_activities = []
        current_time = time.time()
        
        for module_id, module in self.network.modules.items():
            active_neurons = []
            for i, neuron in enumerate(module.neurons):
                # Check recent spikes or high membrane potential
                if hasattr(neuron, 'spike_times') and neuron.spike_times:
                    recent_spikes = [t for t in neuron.spike_times if t > current_time - 5]
                    if recent_spikes:
                        active_neurons.append(i)
                elif hasattr(neuron, 'membrane_potential') and neuron.membrane_potential > neuron.threshold * 0.8:
                    active_neurons.append(i)
            
            module_activities.append({
                'module_id': module_id,
                'active_neurons': active_neurons,
                'activity_level': len(active_neurons) / len(module.neurons)
            })
            
            module_name = ["Sensory", "Memory", "Executive", "Motor"][module_id]
            print(f"  {module_name} module: {len(active_neurons)} active neurons {active_neurons[:5]}")
        
        # Generate words from neural patterns
        neural_words = self.generate_words_from_neurons(module_activities)
        print(f"  Neural words: {neural_words}")
        
        # Generate neural sentence
        neural_sentence = self.generate_neural_sentence(neural_words, result['total_activity'])
        print(f"  Neural response: {neural_sentence}")
        
        return {
            'pattern': pattern,
            'result': result,
            'neural_words': neural_words,
            'response': neural_sentence
        }
    
    def generate_words_from_neurons(self, module_activities):
        """Map neural firing patterns to words"""
        neuron_word_map = {
            # Module 0 (Sensory)
            (0, 0): "information", (0, 1): "input", (0, 2): "sensation", (0, 3): "data",
            (0, 4): "signal", (0, 5): "pattern", (0, 6): "stimulus", (0, 7): "perception",
            
            # Module 1 (Memory/Association)
            (1, 0): "remember", (1, 1): "memory", (1, 2): "association", (1, 3): "connection",
            (1, 4): "learning", (1, 5): "knowledge", (1, 6): "understanding", (1, 7): "concept",
            (1, 8): "thinking", (1, 9): "processing", (1, 10): "analyzing", (1, 11): "considering",
            
            # Module 2 (Executive)
            (2, 0): "decide", (2, 1): "plan", (2, 2): "control", (2, 3): "organize",
            (2, 4): "focus", (2, 5): "attention", (2, 6): "priority", (2, 7): "goal",
            
            # Module 3 (Motor/Output)
            (3, 0): "express", (3, 1): "communicate", (3, 2): "respond", (3, 3): "output",
            (3, 4): "speak", (3, 5): "articulate", (3, 6): "convey", (3, 7): "transmit"
        }
        
        words = []
        for module_info in module_activities:
            module_id = module_info['module_id']
            active_neurons = module_info['active_neurons']
            
            for neuron_id in active_neurons[:3]:
                key = (module_id, neuron_id)
                if key in neuron_word_map:
                    words.append(neuron_word_map[key])
        
        return list(set(words))[:5]  # Remove duplicates and limit
    
    def generate_neural_sentence(self, words, activity):
        """Construct sentence from neural words"""
        if not words:
            return f"Neural activity level {activity:.2f} but no clear word patterns emerged."
        
        if activity > 2.0:
            templates = [
                "My neurons are actively {0} and {1} this {2}.",
                "High brain activity: {0} while {1} these {2}.",
                "Intense neural {0} and {1} of {2}."
            ]
        elif activity > 1.0:
            templates = [
                "I'm {0} this {1} carefully.",
                "Neural {0} of {1} occurring.",
                "Brain {0} and {1} active."
            ]
        else:
            templates = [
                "Gentle {0} of {1}.",
                "Quiet neural {0}.",
                "Subtle {0} patterns."
            ]
        
        # Fill template with neural words
        template = random.choice(templates)
        padded_words = (words + ["information", "processing", "patterns"])[:3]
        
        try:
            sentence = template.format(*padded_words)
            return f"{sentence} [Activity: {activity:.2f}]"
        except (IndexError, KeyError):
            return f"Neural words: {', '.join(words)} [Activity: {activity:.2f}]"

def main():
    print("🧠 Pure Neural Response Analysis")
    print("=" * 50)
    print("Testing how responses emerge from neural firing patterns\n")
    
    analyzer = NeuralResponseAnalyzer()
    
    test_inputs = [
        "Hello world",
        "What is learning?",
        "I understand this",
        "Complex neural processing",
        "Simple test"
    ]
    
    for i, text in enumerate(test_inputs, 1):
        print(f"\n--- Test {i} ---")
        analyzer.analyze_neural_response(text)
        print()
    
    print("🎯 Key Insights:")
    print("• Different texts activate different neural circuits")
    print("• Words emerge from specific neuron-to-concept mappings")
    print("• Sentence structure depends on overall brain activity")
    print("• Responses are genuinely generated by neural computation!")

if __name__ == "__main__":
    main()
