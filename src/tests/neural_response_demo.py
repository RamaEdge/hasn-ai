#!/usr/bin/env python3
"""
Neural Response Generation Demo
Demonstrates how responses are generated directly from neural firing patterns
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from brain_ai_interactive import InteractiveBrainAI

def demo_neural_response_generation():
    """Show how responses are generated from actual neural patterns"""
    print("🧠 Neural Response Generation Demo")
    print("=" * 50)
    print("This demo shows how responses emerge from actual neural firing patterns")
    print("rather than being pre-programmed.\n")
    
    # Initialize brain AI
    brain_ai = InteractiveBrainAI()
    
    # Test different inputs to show varied neural responses
    test_inputs = [
        "Hello, how are you?",
        "What is learning?", 
        "I want to understand neural networks",
        "This is very complex information",
        "Simple test",
        "Can you remember our conversation?",
        "I feel excited about this technology"
    ]
    
    print("🔬 Testing Neural Response Generation:")
    print("-" * 40)
    
    for i, input_text in enumerate(test_inputs, 1):
        print(f"\n{i}. Input: '{input_text}'")
        
        # Convert to neural pattern
        pattern = brain_ai.text_to_pattern(input_text)
        print(f"   Neural pattern: {pattern}")
        
        # Process through network
        result = brain_ai.network.step(pattern)
        
        # Show neural state
        print("   Network state:")
        print(f"     Activity: {result['total_activity']:.3f}")
        print(f"     Memory: {result['memory_size']}/7")
        print(f"     Attention: {[f'{a:.2f}' for a in result['attention']]}")
        
        # Generate neural response
        response = brain_ai.generate_response(result)
        print(f"   Neural response: {response}")
        
        # Show how response was constructed
        print("   Response source: Direct neural pattern analysis")
        
    print("\n" + "=" * 50)
    print("🎯 Key Points:")
    print("• Responses emerge from actual neural firing patterns")
    print("• Different inputs create different neural states") 
    print("• Response content depends on which neurons are active")
    print("• Memory and attention influence response generation")
    print("• No pre-written response templates - all neural!")
    
    # Show detailed neural analysis for one example
    print("\n🔍 Detailed Neural Analysis Example:")
    print("-" * 30)
    
    example_input = "I want to learn about consciousness"
    print(f"Input: '{example_input}'")
    
    pattern = brain_ai.text_to_pattern(example_input)
    result = brain_ai.network.step(pattern)
    
    # Extract detailed neural information 
    print("\nNeural Pattern Analysis:")
    for module_id, activations in pattern.items():
        module_name = ["Sensory", "Memory", "Executive", "Motor"][module_id]
        print(f"  Module {module_id} ({module_name}): {len(activations)} neurons active")
        print(f"    Active neurons: {list(activations.keys())}")
    
    # Show how words are generated from neural patterns
    module_activities = []
    for module_id, module in brain_ai.network.modules.items():
        active_neurons = []
        for i, neuron in enumerate(module.neurons):
            if hasattr(neuron, 'membrane_potential') and neuron.membrane_potential > neuron.threshold * 0.7:
                active_neurons.append(i)
        
        module_activities.append({
            'module_id': module_id,
            'active_neurons': active_neurons,
            'activity_level': len(active_neurons) / len(module.neurons)
        })
    
    neural_words = brain_ai.generate_words_from_neurons(module_activities)
    print("\nNeural Word Generation:")
    print(f"  Words from firing patterns: {neural_words}")
    
    response = brain_ai.generate_response(result)
    print(f"\nFinal Neural Response: {response}")
    
    print("\n✨ This demonstrates true neural response generation!")
    print("The AI's words emerge from actual brain-like neural activity.")

if __name__ == "__main__":
    demo_neural_response_generation()
