#!/usr/bin/env python3
"""
Interactive Brain AI Training & Chat Interface
A comprehensive interface for training and interacting with the brain-inspired neural network
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from simple_brain_demo import SimpleBrainNetwork
import time
import random
import numpy as np

class InteractiveBrainAI:
    def __init__(self, module_sizes=None):
        """Initialize the brain-inspired AI system"""
        if module_sizes is None:
            module_sizes = [30, 25, 20, 15]
        self.network = SimpleBrainNetwork(module_sizes)
        self.conversation_memory = []
        self.learned_patterns = {}
        self.training_history = []
        
        print("🧠 Brain AI Initialized!")
        print(f"   Modules: {module_sizes}")
        print(f"   Total neurons: {sum(module_sizes)}")
        print("   Memory capacity: 7 items")
        
    def text_to_pattern(self, text):
        """Convert text input to neural activation pattern"""
        pattern = {}
        text = text.lower().strip()
        
        # Length-based encoding (sensory processing)
        text_len = len(text)
        if text_len > 0:
            neurons_to_activate = min(text_len // 2, 15)
            pattern[0] = {i: True for i in range(neurons_to_activate)}
        
        # Content-based encoding (memory/association)
        content_neurons = []
        
        # Question indicators
        if any(q_word in text for q_word in ['what', 'how', 'why', 'when', 'where', '?']):
            content_neurons.extend(range(5, 12))
            
        # Emotional content
        if any(e_word in text for e_word in ['good', 'bad', 'happy', 'sad', 'love', 'hate']):
            content_neurons.extend(range(15, 20))
            
        # Learning/memory keywords
        if any(l_word in text for l_word in ['learn', 'remember', 'know', 'think', 'understand']):
            content_neurons.extend(range(8, 15))
            
        # Action/motor keywords
        if any(a_word in text for a_word in ['do', 'make', 'create', 'build', 'go', 'move']):
            content_neurons.extend(range(12, 18))
            
        if content_neurons:
            pattern[1] = {i: True for i in set(content_neurons[:10])}  # Limit to 10 neurons
            
        # Word-specific patterns (executive processing)
        word_hash = hash(text) % 20
        pattern[2] = {i: True for i in range(word_hash % 5, word_hash % 5 + 3)}
        
        # Context from recent conversation (motor/output)
        if len(self.conversation_memory) > 0:
            recent_context = len(self.conversation_memory) % 10
            pattern[3] = {i: True for i in range(recent_context, recent_context + 2)}
            
        return pattern
    
    def generate_response(self, result):
        """Generate text response directly from neural network activity patterns"""
        activity = result['total_activity']
        memory_size = result['memory_size']
        attention = result['attention']
        
        # Extract neural activity patterns from each module
        module_activities = []
        current_time = time.time()
        
        for module_id, module in self.network.modules.items():
            # Get firing pattern from this module
            active_neurons = []
            for i, neuron in enumerate(module.neurons):
                # Check if neuron fired recently (within last 5 seconds)
                if hasattr(neuron, 'spike_times') and neuron.spike_times:
                    recent_spikes = [t for t in neuron.spike_times if t > current_time - 5]
                    if recent_spikes:
                        active_neurons.append(i)
                elif hasattr(neuron, 'membrane_potential') and neuron.membrane_potential > neuron.threshold * 0.8:
                    # Neuron is close to firing
                    active_neurons.append(i)
            
            module_activities.append({
                'module_id': module_id,
                'active_neurons': active_neurons,
                'activity_level': len(active_neurons) / len(module.neurons)
            })
        
        # Generate response based on actual neural patterns
        response_components = []
        
        # Try direct word generation from neural patterns
        neural_words = self.generate_words_from_neurons(module_activities)
        if len(neural_words) >= 2:
            # Use neural word generation
            neural_response = self.generate_neural_sentence(neural_words, activity)
            pattern_info = f" [Neural words: {', '.join(neural_words[:3])}, Activity: {activity:.2f}]"
            return neural_response + pattern_info
        
        # Module 0 (Sensory) - determines response intensity
        sensory_activity = module_activities[0]['activity_level'] if len(module_activities) > 0 else 0
        if sensory_activity > 0.7:
            response_components.append("intense")
        elif sensory_activity > 0.4:
            response_components.append("moderate")
        else:
            response_components.append("subtle")
        
        # Module 1 (Memory/Association) - determines content type
        if len(module_activities) > 1:
            memory_neurons = module_activities[1]['active_neurons']
            if any(n in range(0, 5) for n in memory_neurons):
                response_components.append("factual")
            elif any(n in range(5, 10) for n in memory_neurons):
                response_components.append("questioning")
            elif any(n in range(10, 15) for n in memory_neurons):
                response_components.append("emotional")
            else:
                response_components.append("analytical")
        
        # Module 2 (Executive) - determines response structure
        if len(module_activities) > 2:
            exec_neurons = module_activities[2]['active_neurons']
            if len(exec_neurons) > 10:
                response_components.append("complex")
            elif len(exec_neurons) > 5:
                response_components.append("structured")
            else:
                response_components.append("simple")
        
        # Module 3 (Motor/Output) - determines response style
        if len(module_activities) > 3:
            motor_neurons = module_activities[3]['active_neurons']
            if any(n in range(0, 3) for n in motor_neurons):
                response_components.append("confident")
            elif any(n in range(3, 6) for n in motor_neurons):
                response_components.append("uncertain")
            else:
                response_components.append("neutral")
        
        # Memory influence
        memory_influence = ""
        if memory_size > 6:
            memory_influence = "with rich contextual connections"
        elif memory_size > 4:
            memory_influence = "building on previous patterns"
        elif memory_size > 2:
            memory_influence = "making new associations"
        else:
            memory_influence = "processing fresh information"
        
        # Attention influence
        max_attention = max(attention) if attention else 0
        min_attention = min(attention) if attention else 0
        attention_focus = max_attention - min_attention
        
        if attention_focus > 0.3:
            attention_state = "with sharp focus"
        elif attention_focus > 0.15:
            attention_state = "with selective attention"
        else:
            attention_state = "with distributed processing"
        
        # Construct response from neural patterns
        response = self._construct_neural_response(
            response_components, 
            memory_influence, 
            attention_state, 
            activity
        )
        
        return response
    
    def _construct_neural_response(self, components, memory_influence, attention_state, activity):
        """Construct response text from neural activity components"""
        
        # Base response templates organized by neural pattern combinations
        response_templates = {
            # High activity patterns
            ("intense", "questioning", "complex", "confident"): [
                "My neural networks are highly active - I'm processing this with {memory_influence} and {attention_state}. The patterns suggest deep engagement.",
                "Strong activation across multiple brain regions - this is creating rich neural dynamics {memory_influence} {attention_state}.",
                "High-intensity processing detected - my cognitive systems are fully engaged {memory_influence} {attention_state}."
            ],
            
            # Moderate activity patterns  
            ("moderate", "analytical", "structured", "neutral"): [
                "I'm analyzing this systematically {memory_influence} {attention_state}. The neural patterns show balanced processing.",
                "My brain regions are working in coordination {memory_influence} {attention_state} to understand this.",
                "Structured neural activity emerging - processing this {memory_influence} {attention_state}."
            ],
            
            # Low activity patterns
            ("subtle", "factual", "simple", "uncertain"): [
                "Gentle neural activation detected - I'm quietly processing this {memory_influence} {attention_state}.",
                "Minimal but focused brain activity - considering this {memory_influence} {attention_state}.",
                "Subtle neural patterns forming - learning from this {memory_influence} {attention_state}."
            ],
            
            # Emotional patterns
            ("moderate", "emotional", "complex", "confident"): [
                "My emotional processing centers are active {memory_influence} {attention_state}. This resonates deeply.",
                "Neural patterns suggest strong emotional engagement {memory_influence} {attention_state}.",
                "Emotional and cognitive networks interacting {memory_influence} {attention_state}."
            ],
            
            # Question patterns
            ("intense", "questioning", "structured", "uncertain"): [
                "My inquiry networks are firing actively {memory_influence} {attention_state}. This raises interesting questions.",
                "Question-processing neural circuits engaged {memory_influence} {attention_state}. I'm curious about this.",
                "Interrogative neural patterns detected {memory_influence} {attention_state}. This sparks investigation."
            ]
        }
        
        # Find best matching template
        component_key = tuple(components[:4]) if len(components) >= 4 else tuple(components + ["neutral"] * (4 - len(components)))
        
        if component_key in response_templates:
            templates = response_templates[component_key]
        else:
            # Fallback to activity-based templates
            if activity > 2.0:
                templates = [
                    "High neural activity detected {memory_influence} {attention_state}. This is stimulating complex brain patterns.",
                    "Intense cognitive processing underway {memory_influence} {attention_state}.",
                    "My neural networks are buzzing with activity {memory_influence} {attention_state}."
                ]
            elif activity > 1.0:
                templates = [
                    "Moderate brain activity observed {memory_influence} {attention_state}. Processing this thoughtfully.",
                    "Balanced neural response emerging {memory_influence} {attention_state}.",
                    "My cognitive systems are engaged {memory_influence} {attention_state}."
                ]
            else:
                templates = [
                    "Quiet neural processing {memory_influence} {attention_state}. Considering this carefully.",
                    "Gentle brain activity {memory_influence} {attention_state}. Taking this in slowly.",
                    "Subtle neural patterns {memory_influence} {attention_state}. Learning gradually."
                ]
        
        # Select template and fill in dynamic components
        selected_template = random.choice(templates)
        response = selected_template.format(
            memory_influence=memory_influence,
            attention_state=attention_state
        )
        
        # Add neural pattern details for transparency
        pattern_info = f" [Activity: {activity:.2f}, Pattern: {'-'.join(components[:3])}]"
        
        return response + pattern_info

    def generate_words_from_neurons(self, module_activities):
        """Generate specific words directly from neural firing patterns"""
        words = []
        
        # Map neural patterns to semantic concepts
        neuron_word_map = {
            # Module 0 (Sensory) - Basic concepts
            (0, 0): "information", (0, 1): "input", (0, 2): "sensation", (0, 3): "data",
            (0, 4): "signal", (0, 5): "pattern", (0, 6): "stimulus", (0, 7): "perception",
            
            # Module 1 (Memory/Association) - Cognitive concepts  
            (1, 0): "remember", (1, 1): "memory", (1, 2): "association", (1, 3): "connection",
            (1, 4): "learning", (1, 5): "knowledge", (1, 6): "understanding", (1, 7): "concept",
            (1, 8): "thinking", (1, 9): "processing", (1, 10): "analyzing", (1, 11): "considering",
            
            # Module 2 (Executive) - Action concepts
            (2, 0): "decide", (2, 1): "plan", (2, 2): "control", (2, 3): "organize",
            (2, 4): "focus", (2, 5): "attention", (2, 6): "priority", (2, 7): "goal",
            (2, 8): "strategy", (2, 9): "execute", (2, 10): "manage", (2, 11): "coordinate",
            
            # Module 3 (Motor/Output) - Expression concepts
            (3, 0): "express", (3, 1): "communicate", (3, 2): "respond", (3, 3): "output",
            (3, 4): "speak", (3, 5): "articulate", (3, 6): "convey", (3, 7): "transmit"
        }
        
        # Extract words from active neurons
        for module_info in module_activities:
            module_id = module_info['module_id']
            active_neurons = module_info['active_neurons']
            
            for neuron_id in active_neurons[:3]:  # Limit to top 3 active neurons per module
                key = (module_id, neuron_id)
                if key in neuron_word_map:
                    words.append(neuron_word_map[key])
        
        return words[:5]  # Return top 5 words
    
    def generate_neural_sentence(self, words, activity):
        """Construct sentence using neural-generated words"""
        if not words:
            return "My neural patterns are active but no clear words emerge."
        
        # Sentence structures based on activity level
        if activity > 2.0:
            templates = [
                "I'm actively {0} and {1} while {2} this {3}.",
                "My brain is {0} with {1} as I {2} these {3}.",
                "Intense {0} and {1} occurring while I {2} this {3}."
            ]
        elif activity > 1.0:
            templates = [
                "I'm {0} this and {1} the {2} carefully.",
                "My mind is {0} while {1} these {2}.",
                "Currently {0} and {1} this {2}."
            ]
        else:
            templates = [
                "Gently {0} this {1}.",
                "Quietly {0} and {1}.",
                "Softly {0} these {1}."
            ]
        
        # Fill template with neural-generated words
        template = random.choice(templates)
        try:
            # Pad words list if needed
            padded_words = (words + ["concept", "information", "pattern", "data"])[:4]
            sentence = template.format(*padded_words)
            return sentence
        except (IndexError, KeyError):
            return f"Neural activity generated: {', '.join(words)}"
    
    def learn_pattern(self, name, text):
        """Explicitly teach the AI a pattern"""
        pattern = self.text_to_pattern(text)
        
        print(f"📚 Teaching pattern '{name}'...")
        
        # Multiple training iterations
        improvements = []
        for _ in range(5):
            result = self.network.step(pattern)
            improvements.append(result['total_activity'])
            
        self.learned_patterns[name] = {
            'pattern': pattern,
            'text': text,
            'final_activity': improvements[-1],
            'learning_curve': improvements
        }
        
        avg_improvement = np.mean(improvements)
        print(f"   ✅ Pattern learned! Average activity: {avg_improvement:.3f}")
        return avg_improvement
    
    def chat_mode(self):
        """Interactive chat interface"""
        print("\n💬 Chat Mode - Talk with Brain AI")
        print("   Type 'quit' to exit, 'stats' for network status, 'teach <name>: <pattern>' to teach")
        print("   Example: teach greeting: hello how are you\n")
        
        while True:
            try:
                user_input = input("You: ").strip()
                
                if user_input.lower() == 'quit':
                    print("🧠 Thanks for chatting! Goodbye!")
                    break
                    
                elif user_input.lower() == 'stats':
                    self.show_stats()
                    continue
                    
                elif user_input.lower().startswith('teach '):
                    # Parse teaching command: "teach name: pattern"
                    try:
                        teach_content = user_input[6:]  # Remove "teach "
                        if ':' in teach_content:
                            name, pattern_text = teach_content.split(':', 1)
                            name = name.strip()
                            pattern_text = pattern_text.strip()
                            self.learn_pattern(name, pattern_text)
                        else:
                            print("❌ Format: teach <name>: <pattern>")
                    except ValueError as e:
                        print(f"❌ Teaching error: {e}")
                    continue
                
                # Regular conversation
                pattern = self.text_to_pattern(user_input)
                result = self.network.step(pattern)
                
                # Generate response
                response = self.generate_response(result)
                print(f"AI: {response}")
                
                # Store conversation
                self.conversation_memory.append(('user', user_input))
                self.conversation_memory.append(('ai', response))
                
                # Keep memory manageable
                if len(self.conversation_memory) > 20:
                    self.conversation_memory = self.conversation_memory[-20:]
                
                # Show brief activity info
                print(f"   [Activity: {result['total_activity']:.2f}, Memory: {result['memory_size']}/7]")
                
            except (EOFError, KeyboardInterrupt):
                print("\n🧠 Goodbye!")
                break
            except ValueError as e:
                print(f"❌ Input error: {e}")
    
    def training_mode(self):
        """Structured training interface"""
        print("\n🎓 Training Mode")
        print("1. Basic Pattern Training")
        print("2. Sequence Training") 
        print("3. Custom Pattern Training")
        print("4. Random Pattern Training")
        print("5. Return to main menu")
        
        choice = input("Choose training type (1-5): ").strip()
        
        if choice == '1':
            self.basic_pattern_training()
        elif choice == '2':
            self.sequence_training()
        elif choice == '3':
            self.custom_pattern_training()
        elif choice == '4':
            self.random_pattern_training()
        else:
            return
    
    def basic_pattern_training(self):
        """Train on predefined patterns"""
        patterns = {
            "greeting": "hello how are you doing today",
            "question": "what is the meaning of this concept",
            "emotion": "I feel happy and excited about learning",
            "memory": "remember this important information for later",
            "action": "let's create something new and innovative"
        }
        
        print("\n📚 Basic Pattern Training")
        epochs = int(input("Number of training epochs (1-20): ") or "5")
        
        for epoch in range(epochs):
            print(f"\nEpoch {epoch + 1}/{epochs}")
            epoch_activities = []
            
            for name, text in patterns.items():
                pattern = self.text_to_pattern(text)
                result = self.network.step(pattern)
                activity = result['total_activity']
                epoch_activities.append(activity)
                print(f"  {name}: {activity:.3f}")
            
            avg_activity = np.mean(epoch_activities)
            print(f"  Average activity: {avg_activity:.3f}")
        
        print("✅ Basic training complete!")
    
    def sequence_training(self):
        """Train on temporal sequences"""
        sequences = [
            ["hello", "how are you", "I am fine", "goodbye"],
            ["what is", "the meaning", "of this", "concept"],
            ["I want to", "learn more", "about this", "topic"]
        ]
        
        print("\n🔄 Sequence Training")
        cycles = int(input("Number of sequence cycles (1-10): ") or "3")
        
        for cycle in range(cycles):
            print(f"\nCycle {cycle + 1}/{cycles}")
            
            for seq_idx, sequence in enumerate(sequences):
                print(f"  Sequence {seq_idx + 1}:")
                
                for step_idx, text in enumerate(sequence):
                    pattern = self.text_to_pattern(text)
                    result = self.network.step(pattern)
                    print(f"    Step {step_idx + 1} '{text}': {result['total_activity']:.3f}")
                    time.sleep(0.1)  # Brief pause between sequence steps
        
        print("✅ Sequence training complete!")
    
    def custom_pattern_training(self):
        """Train on user-defined patterns"""
        print("\n✏️  Custom Pattern Training")
        print("Enter patterns to train on (one per line, empty line to start training):")
        
        patterns = []
        while True:
            pattern_text = input(f"Pattern {len(patterns) + 1}: ").strip()
            if not pattern_text:
                break
            patterns.append(pattern_text)
        
        if not patterns:
            print("No patterns entered.")
            return
        
        iterations = int(input("Training iterations per pattern (1-20): ") or "5")
        
        print(f"\n🎯 Training on {len(patterns)} custom patterns...")
        
        for iteration in range(iterations):
            print(f"\nIteration {iteration + 1}/{iterations}")
            
            for idx, text in enumerate(patterns):
                pattern = self.text_to_pattern(text)
                result = self.network.step(pattern)
                print(f"  Pattern {idx + 1}: {result['total_activity']:.3f}")
        
        print("✅ Custom training complete!")
    
    def random_pattern_training(self):
        """Train on random patterns for exploration"""
        print("\n🎲 Random Pattern Training")
        
        num_patterns = int(input("Number of random patterns (1-50): ") or "10")
        iterations = int(input("Iterations per pattern (1-10): ") or "3")
        
        print(f"\n🔀 Generating and training on {num_patterns} random patterns...")
        
        for pattern_idx in range(num_patterns):
            # Generate random pattern
            pattern = {}
            for module_idx in range(len(self.network.modules)):
                if random.random() > 0.3:  # 70% chance to activate module
                    num_neurons = random.randint(1, 8)
                    max_neuron = len(self.network.modules[module_idx].neurons)
                    neurons = random.sample(range(max_neuron), min(num_neurons, max_neuron))
                    pattern[module_idx] = {n: True for n in neurons}
            
            print(f"\nRandom Pattern {pattern_idx + 1}:")
            activities = []
            
            for _ in range(iterations):
                result = self.network.step(pattern)
                activities.append(result['total_activity'])
            
            avg_activity = np.mean(activities)
            print(f"  Average activity: {avg_activity:.3f}")
            print(f"  Pattern: {pattern}")
        
        print("✅ Random training complete!")
    
    def show_stats(self):
        """Display current network statistics"""
        print("\n📊 Brain AI Statistics")
        print("=" * 40)
        
        # Basic network info
        total_neurons = sum(len(module.neurons) for module in self.network.modules.values())
        print(f"Total neurons: {total_neurons}")
        print(f"Modules: {len(self.network.modules)}")
        print("Memory capacity: 7 items")
        
        # Current state
        dummy_pattern = {0: {0: True}}  # Minimal pattern for testing
        result = self.network.step(dummy_pattern)
        
        print("\nCurrent State:")
        print(f"  Activity level: {result['total_activity']:.3f}")
        print(f"  Memory usage: {result['memory_size']}/7")
        print(f"  Attention distribution: {[f'{a:.3f}' for a in result['attention']]}")
        
        # Learning statistics
        if self.learned_patterns:
            print(f"\nLearned Patterns: {len(self.learned_patterns)}")
            for name, info in self.learned_patterns.items():
                print(f"  '{name}': activity {info['final_activity']:.3f}")
        
        # Conversation statistics
        print(f"\nConversation history: {len(self.conversation_memory)} messages")
        
        print("=" * 40)
    
    def main_menu(self):
        """Main interactive menu"""
        print("\n🧠 Brain AI Interactive System")
        print("=============================")
        
        while True:
            print("\nChoose an option:")
            print("1. 💬 Chat Mode - Interactive conversation")
            print("2. 🎓 Training Mode - Structured learning")
            print("3. 📊 Show Statistics - Network status")
            print("4. 🔬 Quick Demo - See the brain in action")
            print("5. 🚪 Exit")
            
            choice = input("\nEnter choice (1-5): ").strip()
            
            if choice == '1':
                self.chat_mode()
            elif choice == '2':
                self.training_mode()
            elif choice == '3':
                self.show_stats()
            elif choice == '4':
                self.quick_demo()
            elif choice == '5':
                print("🧠 Goodbye! Thanks for training with Brain AI!")
                break
            else:
                print("❌ Invalid choice. Please enter 1-5.")
    
    def quick_demo(self):
        """Quick demonstration of brain capabilities"""
        print("\n🔬 Quick Brain Demo")
        print("Demonstrating pattern learning and recognition...")
        
        # Demo patterns
        demo_patterns = {
            "hello": "hello world how are you",
            "learn": "I want to learn new things",
            "think": "let me think about this problem"
        }
        
        print("\n1. Teaching patterns:")
        for name, text in demo_patterns.items():
            pattern = self.text_to_pattern(text)
            activities = []
            for _ in range(3):
                result = self.network.step(pattern)
                activities.append(result['total_activity'])
            
            improvement = activities[-1] - activities[0]
            print(f"   '{name}': {activities[0]:.3f} → {activities[-1]:.3f} (change: {improvement:+.3f})")
        
        print("\n2. Testing recognition:")
        test_phrases = ["hello there", "I love learning", "thinking is fun"]
        
        for phrase in test_phrases:
            pattern = self.text_to_pattern(phrase)
            result = self.network.step(pattern)
            print(f"   '{phrase}': activity {result['total_activity']:.3f}")
        
        print("\n3. Current brain state:")
        print(f"   Memory usage: {result['memory_size']}/7")
        print(f"   Attention: {[f'{a:.2f}' for a in result['attention']]}")
        
        print("\n✅ Demo complete! The brain is learning and responding!")


def main():
    """Main entry point"""
    print("🧠 Welcome to Brain AI Interactive System!")
    print("A biologically-inspired neural network with continuous learning")
    print("")
    
    try:
        # Initialize brain AI
        brain_ai = InteractiveBrainAI()
        
        # Run main menu
        brain_ai.main_menu()
        
    except (KeyboardInterrupt, EOFError):
        print("\n🧠 Session interrupted. Goodbye!")
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Please ensure all required modules are available.")
    except RuntimeError as e:
        print(f"❌ Runtime error: {e}")
        print("Please report this issue.")


if __name__ == "__main__":
    main()
