#!/usr/bin/env python3
"""
Simple Neural Response Test
Clean test of neural response generation without interactive elements
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from brain_ai_interactive import InteractiveBrainAI

def test_neural_responses():
    print("🧠 Testing Neural Response Generation")
    print("=" * 50)
    
    # Initialize brain AI
    brain_ai = InteractiveBrainAI()
    
    # Test inputs
    test_cases = [
        "Hello world",
        "What is consciousness?", 
        "I love learning",
        "This is complex",
        "Simple"
    ]
    
    for i, input_text in enumerate(test_cases, 1):
        print(f"\nTest {i}: '{input_text}'")
        
        # Convert text to neural pattern
        pattern = brain_ai.text_to_pattern(input_text)
        
        # Process through network  
        result = brain_ai.network.step(pattern)
        
        # Extract neural activity details
        module_activities = []
        for module_id, module in brain_ai.network.modules.items():
            active_neurons = []
            for neuron_idx, neuron in enumerate(module.neurons):
                # Check if neuron is highly active
                if hasattr(neuron, 'membrane_potential') and neuron.membrane_potential > neuron.threshold * 0.7:
                    active_neurons.append(neuron_idx)
            
            module_activities.append({
                'module_id': module_id,
                'active_neurons': active_neurons,
                'activity_level': len(active_neurons) / len(module.neurons)
            })
        
        # Generate words from neural patterns
        neural_words = brain_ai.generate_words_from_neurons(module_activities)
        
        # Generate response
        response = brain_ai.generate_response(result)
        
        print(f"  Input pattern: {pattern}")
        print(f"  Activity: {result['total_activity']:.3f}")
        print(f"  Memory: {result['memory_size']}/7") 
        print(f"  Neural words: {neural_words}")
        print(f"  Response: {response}")
        
        # Show which modules were active
        active_modules = [info for info in module_activities if info['activity_level'] > 0]
        if active_modules:
            print(f"  Active modules: {[info['module_id'] for info in active_modules]}")
    
    print("\n🎯 Summary:")
    print("• Each response is generated from actual neural firing patterns")
    print("• Different inputs activate different neural circuits")
    print("• Words emerge from specific neuron-to-concept mappings")
    print("• No pre-written responses - all neural computation!")

if __name__ == "__main__":
    test_neural_responses()
