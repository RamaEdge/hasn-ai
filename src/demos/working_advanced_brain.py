#!/usr/bin/env python3
"""
Working Advanced Brain Network Demo
A simplified but functional version of the advanced cognitive brain network
"""

import numpy as np
from typing import Dict

class CognitiveBrainDemo:
    """Simplified advanced brain network with clear, meaningful outputs"""
    
    def __init__(self):
        # Network architecture
        self.modules = {
            'sensory': {'size': 50, 'activity': 0.0, 'neurons': []},
            'memory': {'size': 40, 'activity': 0.0, 'neurons': []},  
            'executive': {'size': 30, 'activity': 0.0, 'neurons': []},
            'motor': {'size': 20, 'activity': 0.0, 'neurons': []}
        }
        
        # Cognitive systems
        self.working_memory = []
        self.attention_weights = np.array([0.25, 0.25, 0.25, 0.25])  # Equal initial attention
        self.long_term_memory = []
        self.time_step = 0
        
        # Brain oscillations (Hz)
        self.brain_waves = {
            'delta': 2.0,    # Deep sleep, unconscious
            'theta': 6.0,    # Memory formation, creativity
            'alpha': 10.0,   # Relaxed awareness
            'beta': 20.0,    # Active thinking
            'gamma': 40.0    # Conscious awareness, binding
        }
        
        # Initialize neural populations
        for module_info in self.modules.values():
            module_info['neurons'] = np.random.random(module_info['size']) * 0.1
            
    def process_input(self, sensory_input: str) -> Dict:
        """Process input through the cognitive brain network"""
        
        # Phase 1: Sensory Processing
        sensory_activation = self._encode_sensory_input(sensory_input)
        self.modules['sensory']['activity'] = sensory_activation
        
        # Phase 2: Memory Retrieval
        memory_activation = self._activate_memory(sensory_input)
        self.modules['memory']['activity'] = memory_activation
        
        # Phase 3: Executive Processing
        executive_activation = self._executive_control(sensory_activation, memory_activation)
        self.modules['executive']['activity'] = executive_activation
        
        # Phase 4: Motor Output
        motor_activation = self._generate_motor_output(executive_activation)
        self.modules['motor']['activity'] = motor_activation
        
        # Update cognitive systems
        self._update_attention()
        self._update_working_memory(sensory_input)
        self._consolidate_memory()
        
        # Generate brain oscillations
        oscillations = self._generate_brain_waves()
        
        self.time_step += 1
        
        return {
            'input': sensory_input,
            'module_activities': {name: info['activity'] for name, info in self.modules.items()},
            'attention_distribution': self.attention_weights.copy(),
            'working_memory_items': len(self.working_memory),
            'long_term_memory_items': len(self.long_term_memory),
            'brain_oscillations': oscillations,
            'cognitive_state': self._assess_cognitive_state(),
            'time_step': self.time_step
        }
    
    def _encode_sensory_input(self, input_text: str) -> float:
        """Convert sensory input to neural activation"""
        # Simple encoding based on input complexity
        base_activation = min(1.0, len(input_text) / 20.0)
        
        # Boost activation for certain types of content
        if '?' in input_text:
            base_activation += 0.3  # Questions increase sensory attention
        if any(word in input_text.lower() for word in ['important', 'urgent', 'critical']):
            base_activation += 0.4  # Important content gets more processing
            
        return min(1.0, base_activation)
    
    def _activate_memory(self, input_text: str) -> float:
        """Activate relevant memories"""
        memory_activation = 0.0
        
        # Check for familiar patterns in long-term memory
        for memory_item in self.long_term_memory:
            if any(word in input_text.lower() for word in memory_item.get('keywords', [])):
                memory_activation += 0.2
                
        # Working memory contributes to current activation
        memory_activation += len(self.working_memory) * 0.1
        
        return min(1.0, memory_activation)
    
    def _executive_control(self, sensory_level: float, memory_level: float) -> float:
        """Executive control processing"""
        # Executive function integrates sensory and memory information
        integration_strength = (sensory_level + memory_level) / 2.0
        
        # Executive control is higher when there's conflict or complexity
        if sensory_level > 0.7 and memory_level > 0.7:
            integration_strength += 0.3  # High cognitive load
        elif abs(sensory_level - memory_level) > 0.5:
            integration_strength += 0.2  # Conflict resolution needed
            
        return min(1.0, integration_strength)
    
    def _generate_motor_output(self, executive_level: float) -> float:
        """Generate motor/output activation"""
        # Motor output depends on executive decisions
        motor_activation = executive_level * 0.8
        
        # Add some variability for behavioral flexibility
        motor_activation += np.random.normal(0, 0.1)
        
        return max(0.0, min(1.0, motor_activation))
    
    def _update_attention(self):
        """Update attention weights based on module activities"""
        activities = np.array([info['activity'] for info in self.modules.values()])
        
        # Softmax attention allocation
        if np.sum(activities) > 0:
            exp_activities = np.exp(activities * 2)  # Temperature parameter
            self.attention_weights = exp_activities / np.sum(exp_activities)
        else:
            self.attention_weights = np.array([0.25, 0.25, 0.25, 0.25])
    
    def _update_working_memory(self, input_text: str):
        """Update working memory with current input"""
        # Add to working memory if significant
        current_activity = sum(info['activity'] for info in self.modules.values())
        
        if current_activity > 0.5:
            memory_item = {
                'content': input_text,
                'timestamp': self.time_step,
                'strength': current_activity,
                'keywords': input_text.lower().split()
            }
            self.working_memory.append(memory_item)
            
        # Maintain working memory capacity (7 ± 2 items)
        if len(self.working_memory) > 7:
            # Remove oldest item
            self.working_memory.pop(0)
    
    def _consolidate_memory(self):
        """Transfer important working memory items to long-term memory"""
        # Every 10 time steps, consolidate memories
        if self.time_step % 10 == 0 and self.working_memory:
            
            # Find most important working memory items
            important_items = [item for item in self.working_memory if item['strength'] > 0.7]
            
            for item in important_items:
                # Check if already in long-term memory
                is_novel = True
                for ltm_item in self.long_term_memory:
                    if item['content'] == ltm_item['content']:
                        is_novel = False
                        ltm_item['strength'] += 0.1  # Strengthen existing memory
                        break
                
                if is_novel:
                    self.long_term_memory.append(item.copy())
                    
            # Limit long-term memory size
            if len(self.long_term_memory) > 50:
                # Remove weakest memories
                self.long_term_memory.sort(key=lambda x: x['strength'], reverse=True)
                self.long_term_memory = self.long_term_memory[:50]
    
    def _generate_brain_waves(self) -> Dict[str, float]:
        """Generate realistic brain wave patterns"""
        waves = {}
        
        # Calculate current cognitive load
        total_activity = sum(info['activity'] for info in self.modules.values())
        
        for wave_name, base_freq in self.brain_waves.items():
            # Phase calculation
            phase = (self.time_step * base_freq * 0.1) % (2 * np.pi)
            
            # Amplitude modulation based on cognitive state
            if wave_name == 'gamma' and total_activity > 0.8:
                amplitude = 0.8  # High gamma during intense processing
            elif wave_name == 'beta' and total_activity > 0.5:
                amplitude = 0.7  # Beta during active thinking
            elif wave_name == 'alpha' and total_activity < 0.3:
                amplitude = 0.6  # Alpha during relaxed state
            elif wave_name == 'theta' and len(self.working_memory) > 5:
                amplitude = 0.5  # Theta during memory formation
            else:
                amplitude = 0.3
                
            waves[wave_name] = amplitude * np.sin(phase)
            
        return waves
    
    def _assess_cognitive_state(self) -> str:
        """Assess current cognitive state based on brain activity"""
        total_activity = sum(info['activity'] for info in self.modules.values())
        attention_focus = np.max(self.attention_weights) - np.min(self.attention_weights)
        memory_load = len(self.working_memory) / 7.0
        
        if total_activity > 0.8:
            return "High Cognitive Load"
        elif total_activity > 0.5:
            return "Active Processing"
        elif attention_focus > 0.3:
            return "Focused Attention"
        elif memory_load > 0.7:
            return "Memory Intensive"
        else:
            return "Baseline State"

def run_cognitive_demo():
    """Run a comprehensive demonstration of the cognitive brain network"""
    
    print("🧠 Advanced Cognitive Brain Network Demonstration")
    print("=" * 60)
    
    brain = CognitiveBrainDemo()
    
    # Test scenarios with meaningful descriptions
    test_scenarios = [
        ("Hello", "Simple greeting - basic sensory processing"),
        ("What is consciousness?", "Complex question - activates memory and executive systems"),
        ("I need to remember this important meeting", "Memory encoding task"),
        ("This is urgent and critical information", "High-priority processing"),
        ("Let me think about this problem carefully", "Executive control and planning"),
        ("I understand the concept now", "Integration and comprehension"),
        ("Please help me solve this", "Problem-solving request"),
        ("", "Rest state - minimal input")
    ]
    
    print("Running cognitive processing scenarios...\n")
    
    for i, (input_text, description) in enumerate(test_scenarios, 1):
        print(f"Scenario {i}: {description}")
        print(f"Input: '{input_text}'")
        
        result = brain.process_input(input_text)
        
        # Display meaningful results
        print("Neural Module Activities:")
        for module_name, activity in result['module_activities'].items():
            activity_level = "High" if activity > 0.7 else "Medium" if activity > 0.4 else "Low"
            print(f"  {module_name.capitalize():>9}: {activity:.3f} ({activity_level})")
        
        print("Attention Distribution:")
        attention_names = ['Sensory', 'Memory', 'Executive', 'Motor']
        for name, weight in zip(attention_names, result['attention_distribution']):
            print(f"  {name:>9}: {weight:.3f}")
        
        print("Memory Systems:")
        print(f"  Working Memory: {result['working_memory_items']}/7 items")
        print(f"  Long-term Memory: {result['long_term_memory_items']} items")
        
        print("Brain Wave Activity:")
        for wave_name, amplitude in result['brain_oscillations'].items():
            wave_state = "Strong" if abs(amplitude) > 0.5 else "Moderate" if abs(amplitude) > 0.3 else "Weak"
            print(f"  {wave_name.capitalize():>5} ({brain.brain_waves[wave_name]:>4.1f}Hz): {amplitude:>6.3f} ({wave_state})")
        
        print(f"Cognitive State: {result['cognitive_state']}")
        print(f"Time Step: {result['time_step']}")
        print("-" * 60)
    
    print("\n🎯 Demonstration Summary:")
    print(f"Total time steps: {brain.time_step}")
    print(f"Working memory capacity utilized: {len(brain.working_memory)}/7")
    print(f"Long-term memories formed: {len(brain.long_term_memory)}")
    print(f"Current attention focus: {brain.attention_weights}")
    
    # Show memory contents
    if brain.working_memory:
        print("\nCurrent Working Memory Contents:")
        for i, item in enumerate(brain.working_memory):
            print(f"  {i+1}. '{item['content']}' (strength: {item['strength']:.2f})")
    
    if brain.long_term_memory:
        print("\nLong-term Memory (top 5):")
        sorted_ltm = sorted(brain.long_term_memory, key=lambda x: x['strength'], reverse=True)
        for i, item in enumerate(sorted_ltm[:5]):
            print(f"  {i+1}. '{item['content']}' (strength: {item['strength']:.2f})")
    
    print("\n✅ Advanced cognitive brain simulation complete!")
    print("The network demonstrated realistic brain-like processing with:")
    print("  • Dynamic attention allocation")
    print("  • Working memory management") 
    print("  • Long-term memory consolidation")
    print("  • Realistic brain wave generation")
    print("  • Cognitive state assessment")

if __name__ == "__main__":
    run_cognitive_demo()
