"""
Interactive Brain-Inspired Neural Network Trainer
Allows continuous training and real-time interaction with the HASN architecture
"""

import numpy as np
import json
import time
import threading
from typing import Dict, List, Optional, Tuple
from collections import defaultdict, deque
import sys
import os

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from simple_brain_demo import SimpleBrainNetwork, SimpleBrainNeuron, SimpleBrainModule


class InteractiveBrainTrainer:
    """
    Interactive trainer for the brain-inspired network with continuous learning
    and real-time question-answering capabilities
    """
    
    def __init__(self, module_sizes: List[int] = [50, 40, 30, 20]):
        self.network = SimpleBrainNetwork(module_sizes)
        self.training_data = []
        self.pattern_memory = {}  # Store learned patterns
        self.concept_associations = defaultdict(list)  # Concept -> pattern mappings
        self.training_active = False
        self.interaction_history = []
        
        # Enhanced learning parameters
        self.learning_rate = 0.02
        self.consolidation_threshold = 0.7
        self.pattern_recognition_threshold = 0.6
        
        # Training metrics
        self.training_metrics = {
            'patterns_learned': 0,
            'concepts_formed': 0,
            'interactions_processed': 0,
            'learning_stability': 0.0
        }
        
    def encode_text_pattern(self, text: str, module_id: int = 0) -> Dict[int, Dict[int, bool]]:
        """
        Encode text into neural activation patterns
        Simple encoding: map characters to neuron activations
        """
        # Convert text to numerical representation
        text_lower = text.lower()
        encoded_pattern = {}
        
        # Map to module neurons based on text characteristics
        max_neurons = len(self.network.modules[module_id].neurons)
        
        # Create sparse activation pattern based on text
        active_neurons = []
        for i, char in enumerate(text_lower[:max_neurons//2]):
            if char.isalnum():
                neuron_id = (ord(char) - ord('a')) % max_neurons
                active_neurons.append(neuron_id)
                
        # Add some pattern for length and structure
        length_neurons = [i for i in range(len(text) % (max_neurons//4))]
        active_neurons.extend(length_neurons)
        
        # Remove duplicates and create activation pattern
        active_neurons = list(set(active_neurons))
        encoded_pattern[module_id] = {neuron_id: True for neuron_id in active_neurons}
        
        return encoded_pattern
        
    def train_on_pattern(self, pattern_name: str, input_pattern: Dict[int, Dict[int, bool]], 
                        expected_response: Optional[str] = None, num_iterations: int = 10):
        """
        Train the network on a specific pattern with optional expected response
        """
        print(f"🎯 Training on pattern: '{pattern_name}'")
        
        # Store pattern in memory
        self.pattern_memory[pattern_name] = {
            'input_pattern': input_pattern,
            'expected_response': expected_response,
            'training_iterations': num_iterations,
            'learned_at': time.time()
        }
        
        # Training loop
        training_results = []
        for iteration in range(num_iterations):
            result = self.network.step(input_pattern)
            training_results.append(result)
            
            # Print progress
            if iteration % (num_iterations // 4) == 0:
                print(f"  Iteration {iteration}: Activity={result['total_activity']:.3f}, "
                      f"Memory={result['memory_size']}")
                      
        # Analyze training effectiveness
        final_activity = np.mean([r['total_activity'] for r in training_results[-5:]])
        stability = 1.0 - np.std([r['total_activity'] for r in training_results[-10:]])
        
        self.training_metrics['patterns_learned'] += 1
        self.training_metrics['learning_stability'] = 0.9 * self.training_metrics['learning_stability'] + 0.1 * stability
        
        print(f"  ✓ Pattern learned! Final activity: {final_activity:.3f}, Stability: {stability:.3f}")
        
        return training_results
        
    def train_concept_association(self, concept: str, examples: List[str], iterations_per_example: int = 5):
        """
        Train the network to associate a concept with multiple examples
        """
        print(f"\n🧠 Training concept association: '{concept}'")
        
        concept_patterns = []
        for example in examples:
            print(f"  Training example: '{example}'")
            
            # Encode example as pattern
            pattern = self.encode_text_pattern(example, module_id=0)
            
            # Train on this pattern
            training_results = self.train_on_pattern(f"{concept}_{example}", pattern, 
                                                   expected_response=concept, 
                                                   num_iterations=iterations_per_example)
            
            # Store association
            self.concept_associations[concept].append({
                'example': example,
                'pattern': pattern,
                'training_results': training_results
            })
            
            concept_patterns.append(pattern)
            
        self.training_metrics['concepts_formed'] += 1
        print(f"  ✓ Concept '{concept}' learned with {len(examples)} examples")
        
        return concept_patterns
        
    def recognize_pattern(self, input_pattern: Dict[int, Dict[int, bool]]) -> Tuple[str, float]:
        """
        Recognize a pattern by comparing with learned patterns
        """
        best_match = None
        best_similarity = 0.0
        
        # Get current network response to input
        current_response = self.network.step(input_pattern)
        current_activity = np.array([current_response['activities'].get(i, 0.0) 
                                   for i in range(len(self.network.modules))])
        
        # Compare with stored patterns
        for pattern_name, pattern_data in self.pattern_memory.items():
            # Run stored pattern through network
            stored_response = self.network.step(pattern_data['input_pattern'])
            stored_activity = np.array([stored_response['activities'].get(i, 0.0) 
                                      for i in range(len(self.network.modules))])
            
            # Calculate similarity
            if np.linalg.norm(stored_activity) > 0 and np.linalg.norm(current_activity) > 0:
                similarity = np.dot(current_activity, stored_activity) / (
                    np.linalg.norm(current_activity) * np.linalg.norm(stored_activity))
                
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_match = pattern_name
                    
        return best_match or "unknown", best_similarity
        
    def ask_question(self, question: str) -> Dict:
        """
        Process a question and generate a response based on learned knowledge
        """
        print(f"\n❓ Processing question: '{question}'")
        
        # Encode question as pattern
        question_pattern = self.encode_text_pattern(question, module_id=0)
        
        # Get network response
        response = self.network.step(question_pattern)
        
        # Try to recognize the pattern
        recognized_pattern, similarity = self.recognize_pattern(question_pattern)
        
        # Generate response based on recognition
        if similarity > self.pattern_recognition_threshold:
            if recognized_pattern in self.pattern_memory:
                expected_response = self.pattern_memory[recognized_pattern].get('expected_response')
                confidence = similarity
            else:
                expected_response = "Pattern recognized but no specific response trained"
                confidence = similarity * 0.7
        else:
            expected_response = "Unknown - this appears to be a new pattern"
            confidence = 0.1
            
        # Record interaction
        interaction_result = {
            'question': question,
            'recognized_pattern': recognized_pattern,
            'similarity': similarity,
            'response': expected_response,
            'confidence': confidence,
            'network_activity': response['activities'],
            'attention_state': response['attention'],
            'memory_state': response['memory_size'],
            'timestamp': time.time()
        }
        
        self.interaction_history.append(interaction_result)
        self.training_metrics['interactions_processed'] += 1
        
        # Print response
        print(f"🤖 Response: {expected_response}")
        print(f"   Confidence: {confidence:.3f}")
        print(f"   Pattern similarity: {similarity:.3f}")
        print(f"   Network activity: {response['total_activity']:.3f}")
        
        return interaction_result
        
    def continuous_learning_mode(self, enable_auto_training: bool = True):
        """
        Enable continuous learning mode where the network adapts in real-time
        """
        print("\n🔄 Entering continuous learning mode...")
        print("Type 'help' for commands, 'quit' to exit")
        
        self.training_active = True
        
        # Start background training thread if enabled
        if enable_auto_training:
            training_thread = threading.Thread(target=self._background_training, daemon=True)
            training_thread.start()
            
        while self.training_active:
            try:
                user_input = input("\n💬 Enter command/question: ").strip()
                
                if user_input.lower() in ['quit', 'exit', 'q']:
                    self.training_active = False
                    break
                    
                elif user_input.lower() == 'help':
                    self._show_help()
                    
                elif user_input.lower() == 'status':
                    self._show_status()
                    
                elif user_input.lower().startswith('train '):
                    self._handle_training_command(user_input[6:])
                    
                elif user_input.lower().startswith('concept '):
                    self._handle_concept_command(user_input[8:])
                    
                elif user_input.lower() == 'memory':
                    self._show_memory_state()
                    
                elif user_input.lower() == 'metrics':
                    self._show_training_metrics()
                    
                else:
                    # Treat as a question
                    self.ask_question(user_input)
                    
            except KeyboardInterrupt:
                print("\n\n🛑 Stopping continuous learning mode...")
                self.training_active = False
                break
                
        print("✓ Continuous learning mode ended")
        
    def _background_training(self):
        """Background training to maintain network plasticity"""
        while self.training_active:
            # Occasionally replay learned patterns to maintain them
            if self.pattern_memory and np.random.random() < 0.1:
                pattern_name = np.random.choice(list(self.pattern_memory.keys()))
                pattern_data = self.pattern_memory[pattern_name]
                self.network.step(pattern_data['input_pattern'])
                
            time.sleep(1)  # Check every second
            
    def _show_help(self):
        """Show available commands"""
        help_text = """
🔧 AVAILABLE COMMANDS:
  help              - Show this help message
  status            - Show network status
  metrics           - Show training metrics
  memory            - Show memory state
  train <text>      - Train network on text pattern
  concept <name> <examples> - Train concept with examples
  quit/exit/q       - Exit continuous learning mode
  
💡 EXAMPLES:
  train hello world
  concept greeting hello,hi,hey
  What is machine learning?
  How does the brain work?
"""
        print(help_text)
        
    def _show_status(self):
        """Show current network status"""
        print(f"\n📊 NETWORK STATUS:")
        print(f"  Modules: {len(self.network.modules)}")
        print(f"  Working Memory: {len(self.network.working_memory)}/7")
        print(f"  Attention Weights: {[f'{w:.3f}' for w in self.network.attention_weights]}")
        print(f"  Time Steps: {self.network.time_step}")
        print(f"  Patterns Learned: {len(self.pattern_memory)}")
        print(f"  Concepts Formed: {len(self.concept_associations)}")
        
    def _show_memory_state(self):
        """Show detailed memory state"""
        print(f"\n🧠 MEMORY STATE:")
        print(f"  Working Memory Size: {len(self.network.working_memory)}/7")
        print(f"  Learned Patterns: {len(self.pattern_memory)}")
        
        if self.pattern_memory:
            print(f"  Recent Patterns:")
            for pattern_name in list(self.pattern_memory.keys())[-5:]:
                data = self.pattern_memory[pattern_name]
                age = time.time() - data['learned_at']
                print(f"    - {pattern_name} (learned {age:.0f}s ago)")
                
        if self.concept_associations:
            print(f"  Learned Concepts:")
            for concept, examples in self.concept_associations.items():
                print(f"    - {concept}: {len(examples)} examples")
                
    def _show_training_metrics(self):
        """Show training metrics"""
        print(f"\n📈 TRAINING METRICS:")
        for metric, value in self.training_metrics.items():
            print(f"  {metric.replace('_', ' ').title()}: {value}")
            
    def _handle_training_command(self, text: str):
        """Handle training command"""
        if not text.strip():
            print("❌ Please provide text to train on: train <text>")
            return
            
        pattern = self.encode_text_pattern(text)
        self.train_on_pattern(f"user_input_{len(self.pattern_memory)}", pattern)
        
    def _handle_concept_command(self, command: str):
        """Handle concept training command"""
        parts = command.split(' ', 1)
        if len(parts) != 2:
            print("❌ Usage: concept <name> <example1,example2,example3>")
            return
            
        concept_name = parts[0]
        examples = [ex.strip() for ex in parts[1].split(',')]
        
        if len(examples) < 2:
            print("❌ Please provide at least 2 examples separated by commas")
            return
            
        self.train_concept_association(concept_name, examples)
        
    def save_training_state(self, filepath: str):
        """Save the current training state"""
        state = {
            'pattern_memory': self.pattern_memory,
            'concept_associations': dict(self.concept_associations),
            'training_metrics': self.training_metrics,
            'interaction_history': self.interaction_history[-50:],  # Last 50 interactions
            'network_state': {
                'attention_weights': self.network.attention_weights.tolist(),
                'working_memory_size': len(self.network.working_memory),
                'time_step': self.network.time_step
            },
            'timestamp': time.time()
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(state, f, indent=2, default=str)
            
        print(f"💾 Training state saved to: {filepath}")
        
    def load_training_state(self, filepath: str):
        """Load a previously saved training state"""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                state = json.load(f)
                
            self.pattern_memory = state.get('pattern_memory', {})
            self.concept_associations = defaultdict(list, state.get('concept_associations', {}))
            self.training_metrics = state.get('training_metrics', self.training_metrics)
            self.interaction_history = state.get('interaction_history', [])
            
            # Restore network state
            network_state = state.get('network_state', {})
            if 'attention_weights' in network_state:
                self.network.attention_weights = np.array(network_state['attention_weights'])
            if 'time_step' in network_state:
                self.network.time_step = network_state['time_step']
                
            print(f"📂 Training state loaded from: {filepath}")
            print(f"   Patterns: {len(self.pattern_memory)}")
            print(f"   Concepts: {len(self.concept_associations)}")
            print(f"   Interactions: {len(self.interaction_history)}")
            
        except FileNotFoundError:
            print(f"❌ File not found: {filepath}")
        except Exception as e:
            print(f"❌ Error loading state: {e}")


def demo_training_scenarios():
    """Demonstrate different training scenarios"""
    
    print("🎓 BRAIN-INSPIRED NEURAL NETWORK TRAINING DEMO")
    print("=" * 60)
    
    # Create trainer
    trainer = InteractiveBrainTrainer([60, 50, 40, 30])
    
    print("\n📚 SCENARIO 1: Basic Pattern Training")
    print("-" * 40)
    
    # Train basic patterns
    greeting_pattern = trainer.encode_text_pattern("hello", module_id=0)
    trainer.train_on_pattern("greeting", greeting_pattern, "Hello! How can I help you?")
    
    farewell_pattern = trainer.encode_text_pattern("goodbye", module_id=0)
    trainer.train_on_pattern("farewell", farewell_pattern, "Goodbye! See you later!")
    
    print("\n🧠 SCENARIO 2: Concept Association Training")
    print("-" * 40)
    
    # Train concept associations
    trainer.train_concept_association("colors", ["red", "blue", "green", "yellow"])
    trainer.train_concept_association("animals", ["cat", "dog", "bird", "fish"])
    trainer.train_concept_association("emotions", ["happy", "sad", "angry", "excited"])
    
    print("\n❓ SCENARIO 3: Question-Answer Testing")
    print("-" * 40)
    
    # Test recognition
    test_questions = [
        "hello",
        "red",
        "cat", 
        "goodbye",
        "purple",  # Not trained
        "elephant"  # Not trained
    ]
    
    for question in test_questions:
        trainer.ask_question(question)
        
    print("\n💾 SCENARIO 4: Save Training State")
    print("-" * 40)
    
    # Save state
    output_dir = "/Users/ravi.chillerega/sources/cde-hack-session/output"
    trainer.save_training_state(f"{output_dir}/brain_training_state.json")
    
    print("\n📊 Final Training Metrics:")
    trainer._show_training_metrics()
    
    return trainer


def main_interactive_demo():
    """Main interactive demonstration"""
    
    print("🚀 INTERACTIVE BRAIN-INSPIRED AI TRAINER")
    print("=" * 60)
    
    print("\nChoose demonstration mode:")
    print("1. Automated training demo")
    print("2. Interactive continuous learning")
    print("3. Load previous training state")
    
    choice = input("\nEnter choice (1-3): ").strip()
    
    if choice == "1":
        trainer = demo_training_scenarios()
        
    elif choice == "2":
        trainer = InteractiveBrainTrainer()
        print("\n🎯 Starting with fresh network...")
        
        # Quick initial training
        print("🚀 Performing initial training...")
        trainer.train_concept_association("greetings", ["hello", "hi", "hey"])
        trainer.train_concept_association("questions", ["what", "how", "why", "when"])
        
        trainer.continuous_learning_mode()
        
    elif choice == "3":
        trainer = InteractiveBrainTrainer()
        state_file = "/Users/ravi.chillerega/sources/cde-hack-session/output/brain_training_state.json"
        trainer.load_training_state(state_file)
        trainer.continuous_learning_mode()
        
    else:
        print("Invalid choice, running automated demo...")
        trainer = demo_training_scenarios()
        
    return trainer


if __name__ == "__main__":
    trainer = main_interactive_demo()
