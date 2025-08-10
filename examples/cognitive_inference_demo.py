#!/usr/bin/env python3
"""
Cognitive Inference Demo - Demonstrates how the brain creates inferences and correlates memories

This demo shows:
1. Episodic memory storage with context
2. Automatic association discovery
3. Inference generation through memory correlation
4. Memory consolidation and strengthening
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from core.cognitive_brain_network import CognitiveBrainNetwork, CognitiveConfig
import numpy as np
import time


def scenario_1_learning_and_inference():
    """Scenario 1: Learning facts and making inferences"""
    print("🧠 SCENARIO 1: Learning Facts and Making Inferences")
    print("=" * 60)
    
    # Create cognitive network
    config = CognitiveConfig(
        learning_rate=0.03,
        learning_probability=0.4,
        max_episodic_memories=100,
        association_strength_threshold=0.25,
        max_inference_depth=4
    )
    
    network = CognitiveBrainNetwork(num_neurons=150, config=config)
    
    print("\n📚 Phase 1: Learning about different concepts...")
    
    # Phase 1: Learn basic facts
    learning_experiences = [
        # Animals and their properties
        ({1: True, 2: True, 10: True}, {"concept": "dog", "properties": ["animal", "mammal", "pet"], "size": "medium"}),
        ({1: True, 2: True, 11: True}, {"concept": "cat", "properties": ["animal", "mammal", "pet"], "size": "small"}),
        ({1: True, 2: True, 12: True}, {"concept": "elephant", "properties": ["animal", "mammal", "wild"], "size": "large"}),
        ({1: True, 2: True, 13: True}, {"concept": "whale", "properties": ["animal", "mammal", "aquatic"], "size": "huge"}),
        
        # Birds
        ({1: True, 3: True, 20: True}, {"concept": "eagle", "properties": ["animal", "bird", "predator"], "habitat": "sky"}),
        ({1: True, 3: True, 21: True}, {"concept": "penguin", "properties": ["animal", "bird", "aquatic"], "habitat": "ice"}),
        
        # Vehicles
        ({4: True, 5: True, 30: True}, {"concept": "car", "properties": ["vehicle", "land", "fast"], "fuel": "gasoline"}),
        ({4: True, 5: True, 31: True}, {"concept": "bicycle", "properties": ["vehicle", "land", "slow"], "fuel": "human"}),
        ({4: True, 5: True, 32: True}, {"concept": "airplane", "properties": ["vehicle", "air", "fast"], "fuel": "jet"}),
        
        # Foods
        ({6: True, 7: True, 40: True}, {"concept": "apple", "properties": ["food", "fruit", "sweet"], "color": "red"}),
        ({6: True, 7: True, 41: True}, {"concept": "carrot", "properties": ["food", "vegetable", "crunchy"], "color": "orange"}),
    ]
    
    for i, (pattern, context) in enumerate(learning_experiences):
        result = network.step_with_cognition(pattern, context)
        if i % 3 == 0:  # Show progress
            print(f"   Learned: {context['concept']} - {', '.join(context['properties'])}")
    
    print(f"\n🔄 Phase 2: Memory consolidation...")
    for _ in range(3):  # Multiple consolidation rounds
        network.consolidate_memories()
    
    print(f"\n🎯 Phase 3: Testing inference generation...")
    
    # Test 1: Query about a new mammal
    print(f"\n🔍 Test 1: What can we infer about a new mammal?")
    mammal_pattern = {1: True, 2: True, 14: True}  # New mammal pattern
    mammal_context = {"concept": "unknown_mammal", "properties": ["animal", "mammal"], "generate_inferences": True}
    
    result = network.step_with_cognition(mammal_pattern, mammal_context)
    
    print(f"   Generated {len(result['inferences'])} inferences:")
    for i, inference in enumerate(result['inferences'][:3]):
        print(f"   {i+1}. Confidence: {inference['confidence']:.3f}")
        chain_concepts = []
        for mem_id in inference['inference_chain'][:3]:
            if mem_id in network.episodic_memories:
                concept = network.episodic_memories[mem_id].context.get('concept', 'unknown')
                chain_concepts.append(concept)
        print(f"      Related to: {' → '.join(chain_concepts)}")
    
    # Test 2: Query about transportation
    print(f"\n🔍 Test 2: What can we infer about transportation?")
    transport_pattern = {4: True, 5: True, 33: True}  # New vehicle pattern
    transport_context = {"concept": "unknown_vehicle", "properties": ["vehicle"], "generate_inferences": True}
    
    result = network.step_with_cognition(transport_pattern, transport_context)
    
    print(f"   Generated {len(result['inferences'])} inferences:")
    for i, inference in enumerate(result['inferences'][:3]):
        print(f"   {i+1}. Confidence: {inference['confidence']:.3f}")
        chain_concepts = []
        for mem_id in inference['inference_chain'][:3]:
            if mem_id in network.episodic_memories:
                concept = network.episodic_memories[mem_id].context.get('concept', 'unknown')
                chain_concepts.append(concept)
        print(f"      Related to: {' → '.join(chain_concepts)}")
    
    # Show cognitive state
    print(f"\n📊 Final Cognitive State:")
    state = network.get_cognitive_state()
    for key, value in state.items():
        if isinstance(value, float):
            print(f"   {key}: {value:.3f}")
        else:
            print(f"   {key}: {value}")
    
    return network


def scenario_2_temporal_correlations():
    """Scenario 2: Learning temporal sequences and making predictions"""
    print(f"\n\n🧠 SCENARIO 2: Temporal Correlations and Predictions")
    print("=" * 60)
    
    config = CognitiveConfig(
        temporal_association_window=5.0,  # 5 second window
        association_strength_threshold=0.2,
        max_inference_depth=3
    )
    
    network = CognitiveBrainNetwork(num_neurons=100, config=config)
    
    print(f"\n📚 Learning temporal sequences...")
    
    # Simulate daily routine sequences
    daily_sequences = [
        # Morning routine
        ({10: True, 11: True}, {"activity": "wake_up", "time": "morning", "sequence": 1}),
        ({12: True, 13: True}, {"activity": "brush_teeth", "time": "morning", "sequence": 2}),
        ({14: True, 15: True}, {"activity": "eat_breakfast", "time": "morning", "sequence": 3}),
        ({16: True, 17: True}, {"activity": "go_to_work", "time": "morning", "sequence": 4}),
        
        # Work routine
        ({20: True, 21: True}, {"activity": "start_computer", "time": "work", "sequence": 1}),
        ({22: True, 23: True}, {"activity": "check_emails", "time": "work", "sequence": 2}),
        ({24: True, 25: True}, {"activity": "attend_meeting", "time": "work", "sequence": 3}),
        
        # Evening routine
        ({30: True, 31: True}, {"activity": "come_home", "time": "evening", "sequence": 1}),
        ({32: True, 33: True}, {"activity": "cook_dinner", "time": "evening", "sequence": 2}),
        ({34: True, 35: True}, {"activity": "watch_tv", "time": "evening", "sequence": 3}),
        ({36: True, 37: True}, {"activity": "go_to_sleep", "time": "evening", "sequence": 4}),
    ]
    
    # Learn sequences with temporal spacing
    for i, (pattern, context) in enumerate(daily_sequences):
        result = network.step_with_cognition(pattern, context)
        time.sleep(0.1)  # Small delay to create temporal associations
        
        if i % 4 == 0:
            print(f"   Learning {context['time']} routine...")
    
    # Consolidate temporal associations
    print(f"\n🔄 Consolidating temporal associations...")
    network.consolidate_memories()
    
    # Test temporal prediction
    print(f"\n🎯 Testing temporal predictions...")
    
    # Query: What comes after waking up?
    wake_pattern = {10: True, 11: True}
    wake_context = {"activity": "wake_up", "time": "morning", "generate_inferences": True}
    
    result = network.step_with_cognition(wake_pattern, wake_context)
    
    print(f"   Query: What typically happens after waking up?")
    print(f"   Generated {len(result['inferences'])} temporal inferences:")
    
    for i, inference in enumerate(result['inferences'][:3]):
        print(f"   {i+1}. Confidence: {inference['confidence']:.3f}")
        activities = []
        for mem_id in inference['inference_chain']:
            if mem_id in network.episodic_memories:
                activity = network.episodic_memories[mem_id].context.get('activity', 'unknown')
                activities.append(activity)
        print(f"      Predicted sequence: {' → '.join(activities[:4])}")
    
    return network


def scenario_3_analogical_reasoning():
    """Scenario 3: Learning analogies and making analogical inferences"""
    print(f"\n\n🧠 SCENARIO 3: Analogical Reasoning")
    print("=" * 60)
    
    config = CognitiveConfig(
        association_strength_threshold=0.15,  # Lower threshold for analogies
        max_inference_depth=4
    )
    
    network = CognitiveBrainNetwork(num_neurons=200, config=config)
    
    print(f"\n📚 Learning analogical relationships...")
    
    # Learn analogical patterns
    analogical_experiences = [
        # Size relationships
        ({1: True, 10: True, 20: True}, {"analogy": "size", "relation": "small_to_large", "example": "mouse_to_elephant"}),
        ({2: True, 11: True, 21: True}, {"analogy": "size", "relation": "small_to_large", "example": "pebble_to_boulder"}),
        ({3: True, 12: True, 22: True}, {"analogy": "size", "relation": "small_to_large", "example": "drop_to_ocean"}),
        
        # Parent-child relationships
        ({4: True, 14: True, 24: True}, {"analogy": "family", "relation": "parent_child", "example": "dog_puppy"}),
        ({5: True, 15: True, 25: True}, {"analogy": "family", "relation": "parent_child", "example": "cat_kitten"}),
        ({6: True, 16: True, 26: True}, {"analogy": "family", "relation": "parent_child", "example": "bird_chick"}),
        
        # Tool-function relationships
        ({7: True, 17: True, 27: True}, {"analogy": "tool", "relation": "tool_function", "example": "hammer_nail"}),
        ({8: True, 18: True, 28: True}, {"analogy": "tool", "relation": "tool_function", "example": "key_lock"}),
        ({9: True, 19: True, 29: True}, {"analogy": "tool", "relation": "tool_function", "example": "brush_paint"}),
        
        # Cause-effect relationships
        ({30: True, 40: True, 50: True}, {"analogy": "cause_effect", "relation": "cause_effect", "example": "rain_wet"}),
        ({31: True, 41: True, 51: True}, {"analogy": "cause_effect", "relation": "cause_effect", "example": "fire_smoke"}),
        ({32: True, 42: True, 52: True}, {"analogy": "cause_effect", "relation": "cause_effect", "example": "wind_movement"}),
    ]
    
    for i, (pattern, context) in enumerate(analogical_experiences):
        result = network.step_with_cognition(pattern, context)
        if i % 3 == 0:
            print(f"   Learning {context['analogy']} analogies...")
    
    # Consolidate analogical knowledge
    print(f"\n🔄 Consolidating analogical knowledge...")
    for _ in range(2):
        network.consolidate_memories()
    
    # Test analogical reasoning
    print(f"\n🎯 Testing analogical reasoning...")
    
    # Query: Complete the analogy - what's the large version of a seed?
    seed_pattern = {1: True, 13: True, 23: True}  # Similar to size patterns
    seed_context = {"analogy": "size", "relation": "small_to_large", "example": "seed_to_?", "generate_inferences": True}
    
    result = network.step_with_cognition(seed_pattern, seed_context)
    
    print(f"   Analogy query: seed is to ? as mouse is to elephant")
    print(f"   Generated {len(result['inferences'])} analogical inferences:")
    
    for i, inference in enumerate(result['inferences'][:3]):
        print(f"   {i+1}. Confidence: {inference['confidence']:.3f}")
        examples = []
        for mem_id in inference['inference_chain'][:3]:
            if mem_id in network.episodic_memories:
                example = network.episodic_memories[mem_id].context.get('example', 'unknown')
                examples.append(example)
        print(f"      Similar patterns: {', '.join(examples)}")
    
    # Show final state
    print(f"\n📊 Final Analogical Knowledge State:")
    state = network.get_cognitive_state()
    print(f"   Total memories: {state['total_memories']}")
    print(f"   Average associations per memory: {state['avg_associations_per_memory']:.2f}")
    print(f"   Total associations: {state['total_associations']}")
    
    return network


def main():
    """Run all cognitive inference scenarios"""
    print("🧠 COGNITIVE BRAIN NETWORK - INFERENCE & MEMORY CORRELATION DEMO")
    print("=" * 70)
    print()
    print("This demo shows how our brain network:")
    print("• Stores episodic memories with rich context")
    print("• Automatically discovers associations between memories")
    print("• Generates inferences through memory correlation")
    print("• Consolidates important memories over time")
    print("• Performs analogical reasoning")
    print()
    
    # Run scenarios
    network1 = scenario_1_learning_and_inference()
    network2 = scenario_2_temporal_correlations()
    network3 = scenario_3_analogical_reasoning()
    
    print(f"\n\n🎉 DEMO COMPLETE!")
    print("=" * 70)
    print()
    print("🧠 KEY INSIGHTS:")
    print("• Our brain network now creates TRUE INFERENCES, not just pattern matching")
    print("• Memories are automatically associated based on similarity and timing")
    print("• The system can make predictions and analogical reasoning")
    print("• Memory consolidation strengthens important associations over time")
    print("• This mimics how biological brains create knowledge and understanding")
    print()
    print("🚀 NEXT STEPS:")
    print("• Integrate with language processing for conversational AI")
    print("• Add emotional context to memories")
    print("• Implement hierarchical memory organization")
    print("• Create visualization tools for memory networks")


if __name__ == "__main__":
    main()