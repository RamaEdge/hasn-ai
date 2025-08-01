"""
Quick Brain Network Training Demo
Shows how to train and interact with the brain-inspired network
"""

import numpy as np
import sys
import os

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from simple_brain_demo import SimpleBrainNetwork


def quick_training_demo():
    """Quick demonstration of training the brain network"""
    
    print("🧠 BRAIN-INSPIRED NEURAL NETWORK TRAINING DEMO")
    print("=" * 55)
    
    # Create network
    print("\n🔧 Creating brain network with 4 modules...")
    network = SimpleBrainNetwork([25, 20, 15, 10])
    
    print(f"✓ Network created:")
    for i, module in network.modules.items():
        print(f"   Module {i}: {len(module.neurons)} neurons")
    
    # Define training patterns
    patterns = {
        "pattern_A": {0: {i: True for i in range(5)}},
        "pattern_B": {1: {i: True for i in range(3, 8)}},
        "pattern_C": {2: {i: True for i in range(7, 12)}},
        "sequence": {0: {i: True for i in [1, 3, 5]}, 1: {i: True for i in [2, 4, 6]}}
    }
    
    print(f"\n🎯 Training on {len(patterns)} different patterns...")
    
    # Training loop
    training_results = {}
    
    for pattern_name, pattern in patterns.items():
        print(f"\n   Training on {pattern_name}:")
        
        pattern_results = []
        
        # Train for 20 iterations
        for iteration in range(20):
            result = network.step(pattern)
            pattern_results.append(result)
            
            if iteration % 5 == 0:
                print(f"     Iter {iteration:2d}: Activity={result['total_activity']:.3f}, "
                      f"Memory={result['memory_size']}, "
                      f"Attention=[{', '.join(f'{w:.2f}' for w in result['attention'])}]")
        
        training_results[pattern_name] = pattern_results
        
        # Calculate learning metrics
        initial_activity = np.mean([r['total_activity'] for r in pattern_results[:5]])
        final_activity = np.mean([r['total_activity'] for r in pattern_results[-5:]])
        learning_improvement = final_activity - initial_activity
        
        print(f"     Learning improvement: {learning_improvement:+.3f}")
    
    print(f"\n🧠 Testing pattern recognition...")
    
    # Test pattern recognition
    test_patterns = [
        ("Familiar pattern A", patterns["pattern_A"]),
        ("Familiar pattern B", patterns["pattern_B"]),
        ("Novel pattern", {0: {i: True for i in [0, 2, 4, 6, 8]}, 
                          3: {i: True for i in range(2)}}),
        ("Empty pattern", {})
    ]
    
    for test_name, test_pattern in test_patterns:
        result = network.step(test_pattern)
        
        print(f"\n   Testing: {test_name}")
        print(f"     Activity: {result['total_activity']:.3f}")
        print(f"     Memory: {result['memory_size']}/7")
        activities_str = [f"{result['activities'].get(i, 0):.2f}" for i in range(4)]
        print(f"     Module activities: {activities_str}")
        
        # Check if pattern is "recognized" (high activity in trained modules)
        max_activity = max(result['activities'].values()) if result['activities'] else 0
        if max_activity > 0.5:
            active_module = max(result['activities'], key=result['activities'].get)
            print(f"     → Pattern recognized in module {active_module}")
        else:
            print(f"     → Novel/unrecognized pattern")
    
    print(f"\n📊 Final network state:")
    print(f"   Total time steps: {network.time_step}")
    print(f"   Working memory: {len(network.working_memory)}/7 capacity")
    print(f"   Attention distribution: {[f'{w:.3f}' for w in network.attention_weights]}")
    
    # Calculate network learning statistics
    all_activities = []
    all_memory_usage = []
    
    for pattern_results in training_results.values():
        all_activities.extend([r['total_activity'] for r in pattern_results])
        all_memory_usage.extend([r['memory_size'] for r in pattern_results])
    
    print(f"\n📈 Learning statistics:")
    print(f"   Average activity: {np.mean(all_activities):.3f}")
    print(f"   Activity std: {np.std(all_activities):.3f}")
    print(f"   Max memory usage: {max(all_memory_usage)}")
    print(f"   Final attention entropy: {calculate_entropy(network.attention_weights):.3f}")
    
    print(f"\n✨ Key brain-like behaviors observed:")
    behaviors = []
    
    if max(all_memory_usage) >= 5:
        behaviors.append("🧠 Active working memory usage")
    
    if np.std(network.attention_weights) > 0.1:
        behaviors.append("🎯 Selective attention formation")
    
    if np.mean(all_activities[-20:]) > np.mean(all_activities[:20:]):
        behaviors.append("📈 Learning-driven activity increase")
    
    if len(network.working_memory) > 0:
        behaviors.append("💾 Pattern memory consolidation")
    
    for behavior in behaviors:
        print(f"   {behavior}")
    
    print(f"\n🎉 Training demonstration complete!")
    print("   The network has learned to process patterns with")
    print("   brain-like temporal dynamics and memory formation.")
    
    return network, training_results


def calculate_entropy(weights):
    """Calculate entropy of attention weights"""
    weights = np.array(weights) + 1e-8  # Avoid log(0)
    weights = weights / np.sum(weights)
    return -np.sum(weights * np.log(weights))


def interactive_question_demo(network):
    """Simple interactive demo for asking questions"""
    
    print(f"\n💬 INTERACTIVE QUESTION MODE")
    print("=" * 40)
    print("You can now 'ask' the trained network questions by")
    print("entering text. The network will process it as a pattern.")
    print("Type 'quit' to exit.\n")
    
    question_count = 0
    
    while True:
        try:
            question = input("❓ Ask a question: ").strip()
            
            if question.lower() in ['quit', 'exit', 'q']:
                break
                
            if not question:
                continue
                
            question_count += 1
            
            # Convert text to simple pattern
            pattern = text_to_pattern(question)
            
            # Get network response
            result = network.step(pattern)
            
            print(f"🤖 Network response:")
            print(f"   Activity level: {result['total_activity']:.3f}")
            print(f"   Most active module: {max(result['activities'], key=result['activities'].get) if result['activities'] else 'None'}")
            print(f"   Memory updated: {'Yes' if result['memory_size'] > len(network.working_memory) - 1 else 'No'}")
            
            # Simple response based on activity
            if result['total_activity'] > 1.0:
                response = "This seems like a familiar pattern!"
            elif result['total_activity'] > 0.5:
                response = "I recognize some aspects of this."
            else:
                response = "This appears to be a new pattern to me."
                
            print(f"   Interpretation: {response}")
            print()
            
        except KeyboardInterrupt:
            break
    
    print(f"✓ Processed {question_count} questions.")


def text_to_pattern(text):
    """Convert text to neural activation pattern"""
    # Simple text encoding
    pattern = {}
    
    # Use text length and characteristics to determine activation
    text_len = len(text)
    has_question = '?' in text
    has_numbers = any(c.isdigit() for c in text)
    has_caps = any(c.isupper() for c in text)
    
    # Module 0: Based on length
    if text_len > 0:
        pattern[0] = {i: True for i in range(min(text_len // 2, 10))}
    
    # Module 1: Question words
    if has_question or any(word in text.lower() for word in ['what', 'how', 'why', 'when', 'where']):
        pattern[1] = {i: True for i in range(3, 8)}
    
    # Module 2: Numbers or complexity
    if has_numbers or text_len > 20:
        pattern[2] = {i: True for i in range(5, 10)}
    
    # Module 3: Emphasis (caps, exclamation)
    if has_caps or '!' in text:
        pattern[3] = {i: True for i in range(2, 6)}
    
    return pattern


if __name__ == "__main__":
    # Run training demo
    network, results = quick_training_demo()
    
    # Optional interactive mode
    print(f"\n🎮 Would you like to try the interactive question mode? (y/n): ", end="")
    try:
        choice = input().strip().lower()
        if choice in ['y', 'yes']:
            interactive_question_demo(network)
    except KeyboardInterrupt:
        pass
    
    print(f"\n🌟 Demo complete! The brain-inspired network is ready for further training.")
