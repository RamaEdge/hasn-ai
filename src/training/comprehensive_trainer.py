"""
Systematic Training Guide for Brain-Inspired Neural Networks
This module provides structured training methods and examples
"""

import numpy as np
import json
import time
from typing import Dict, List, Tuple, Any
from collections import defaultdict

# Import our brain network
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from simple_brain_demo import SimpleBrainNetwork


class BrainNetworkTrainer:
    """
    Comprehensive trainer for brain-inspired neural networks
    Provides multiple training paradigms and evaluation methods
    """
    
    def __init__(self, network: SimpleBrainNetwork):
        self.network = network
        self.training_history = []
        self.learned_patterns = {}
        self.performance_metrics = defaultdict(list)
        
    def supervised_training(self, training_data: List[Tuple[Dict, str]], epochs: int = 10):
        """
        Supervised training with input-output pairs
        
        Args:
            training_data: List of (input_pattern, expected_output) tuples
            epochs: Number of training epochs
        """
        print(f"🎓 Starting supervised training with {len(training_data)} examples for {epochs} epochs")
        
        for epoch in range(epochs):
            epoch_accuracy = 0.0
            epoch_loss = 0.0
            
            print(f"\n📚 Epoch {epoch + 1}/{epochs}")
            
            for i, (input_pattern, expected_output) in enumerate(training_data):
                # Forward pass
                result = self.network.step(input_pattern)
                
                # Calculate "loss" based on activity patterns
                activity_vector = np.array([result['activities'].get(j, 0.0) 
                                          for j in range(len(self.network.modules))])
                
                # Simple target encoding (could be improved)
                target_vector = self._encode_target(expected_output)
                
                # Calculate loss (MSE)
                loss = np.mean((activity_vector - target_vector) ** 2)
                epoch_loss += loss
                
                # Calculate accuracy (similarity threshold)
                similarity = np.dot(activity_vector, target_vector) / (
                    np.linalg.norm(activity_vector) * np.linalg.norm(target_vector) + 1e-8)
                
                if similarity > 0.7:  # Threshold for "correct"
                    epoch_accuracy += 1.0
                    
                # Store pattern for later recognition
                self.learned_patterns[f"pattern_{len(self.learned_patterns)}"] = {
                    'input': input_pattern,
                    'expected_output': expected_output,
                    'learned_activity': activity_vector,
                    'epoch': epoch
                }
                
            # Calculate epoch metrics
            epoch_accuracy /= len(training_data)
            epoch_loss /= len(training_data)
            
            self.performance_metrics['accuracy'].append(epoch_accuracy)
            self.performance_metrics['loss'].append(epoch_loss)
            
            print(f"   Accuracy: {epoch_accuracy:.3f}, Loss: {epoch_loss:.3f}")
            
        print(f"✅ Supervised training complete!")
        return self.performance_metrics
        
    def unsupervised_training(self, input_patterns: List[Dict], training_steps: int = 1000):
        """
        Unsupervised training - let the network self-organize
        
        Args:
            input_patterns: List of input patterns to learn from
            training_steps: Number of training steps
        """
        print(f"🧠 Starting unsupervised training with {len(input_patterns)} patterns for {training_steps} steps")
        
        pattern_frequencies = defaultdict(int)
        
        for step in range(training_steps):
            # Select random pattern
            pattern = np.random.choice(input_patterns)
            pattern_frequencies[str(pattern)] += 1
            
            # Present to network
            result = self.network.step(pattern)
            
            # Record metrics every 100 steps
            if step % 100 == 0:
                total_activity = result['total_activity']
                memory_usage = result['memory_size']
                attention_entropy = self._calculate_attention_entropy(result['attention'])
                
                self.performance_metrics['activity'].append(total_activity)
                self.performance_metrics['memory_usage'].append(memory_usage)
                self.performance_metrics['attention_entropy'].append(attention_entropy)
                
                print(f"   Step {step}: Activity={total_activity:.3f}, "
                      f"Memory={memory_usage}, Attention_H={attention_entropy:.3f}")
                      
        print(f"✅ Unsupervised training complete!")
        return self.performance_metrics
        
    def reinforcement_training(self, environment_func, num_episodes: int = 100):
        """
        Reinforcement learning training with reward signals
        
        Args:
            environment_func: Function that takes network output and returns (reward, next_state)
            num_episodes: Number of training episodes
        """
        print(f"🎮 Starting reinforcement training for {num_episodes} episodes")
        
        total_rewards = []
        
        for episode in range(num_episodes):
            episode_reward = 0.0
            episode_steps = 0
            
            # Initialize environment
            state = environment_func(None, reset=True)
            
            while episode_steps < 50:  # Max steps per episode
                # Get network response to current state
                result = self.network.step(state)
                
                # Extract action from network activity
                action = self._extract_action(result)
                
                # Get environment response
                reward, next_state, done = environment_func(action)
                episode_reward += reward
                
                # Simple reward-based modulation of learning
                if reward > 0:
                    # Positive reward - strengthen recent patterns
                    self._reinforce_recent_activity(1.2)
                elif reward < 0:
                    # Negative reward - weaken recent patterns
                    self._reinforce_recent_activity(0.8)
                    
                state = next_state
                episode_steps += 1
                
                if done:
                    break
                    
            total_rewards.append(episode_reward)
            
            if episode % 10 == 0:
                avg_reward = np.mean(total_rewards[-10:])
                print(f"   Episode {episode}: Avg Reward (last 10) = {avg_reward:.3f}")
                
        self.performance_metrics['episode_rewards'] = total_rewards
        print(f"✅ Reinforcement training complete!")
        return self.performance_metrics
        
    def continual_learning(self, task_sequence: List[Tuple[str, List]], consolidation_steps: int = 50):
        """
        Continual learning across multiple tasks
        
        Args:
            task_sequence: List of (task_name, training_data) tuples
            consolidation_steps: Steps for memory consolidation between tasks
        """
        print(f"🔄 Starting continual learning across {len(task_sequence)} tasks")
        
        task_performances = {}
        
        for task_id, (task_name, task_data) in enumerate(task_sequence):
            print(f"\n📖 Learning Task {task_id + 1}: {task_name}")
            
            # Train on current task
            task_metrics = self.supervised_training(task_data, epochs=5)
            
            # Memory consolidation phase
            print(f"🧠 Memory consolidation for {consolidation_steps} steps...")
            self._memory_consolidation(consolidation_steps)
            
            # Test on all previous tasks (measure forgetting)
            task_performance = {}
            for prev_task_id, (prev_task_name, prev_task_data) in enumerate(task_sequence[:task_id + 1]):
                accuracy = self._test_task_performance(prev_task_data)
                task_performance[prev_task_name] = accuracy
                print(f"   {prev_task_name}: {accuracy:.3f}")
                
            task_performances[task_name] = task_performance
            
        self.performance_metrics['continual_learning'] = task_performances
        print(f"✅ Continual learning complete!")
        return task_performances
        
    def meta_learning(self, task_distribution: List[List], meta_episodes: int = 20):
        """
        Meta-learning (learning to learn)
        
        Args:
            task_distribution: List of different task datasets
            meta_episodes: Number of meta-learning episodes
        """
        print(f"🚀 Starting meta-learning across {len(task_distribution)} task types for {meta_episodes} episodes")
        
        adaptation_speeds = []
        
        for episode in range(meta_episodes):
            # Sample a task from the distribution
            task_data = np.random.choice(task_distribution)
            
            # Measure adaptation speed
            initial_performance = self._test_task_performance(task_data[:2])  # Test on first 2 examples
            
            # Quick adaptation (few training steps)
            for example in task_data[:5]:  # Train on first 5 examples
                self.network.step(example[0])
                
            # Measure performance after adaptation
            adapted_performance = self._test_task_performance(task_data[5:7])  # Test on next 2
            
            adaptation_speed = adapted_performance - initial_performance
            adaptation_speeds.append(adaptation_speed)
            
            if episode % 5 == 0:
                avg_adaptation = np.mean(adaptation_speeds[-5:])
                print(f"   Episode {episode}: Avg Adaptation Speed = {avg_adaptation:.3f}")
                
        self.performance_metrics['adaptation_speeds'] = adaptation_speeds
        print(f"✅ Meta-learning complete!")
        return adaptation_speeds
        
    def _encode_target(self, target: str) -> np.ndarray:
        """Encode target string as activity vector"""
        # Simple encoding: map string characteristics to module activities
        target_vector = np.zeros(len(self.network.modules))
        
        # Length-based encoding
        target_vector[0] = min(1.0, len(target) / 10.0)
        
        # Character-based encoding
        if any(c.isdigit() for c in target):
            target_vector[1] = 0.8
        if any(c.isupper() for c in target):
            target_vector[2] = 0.7
        if len(target) > 5:
            target_vector[3] = 0.6
            
        return target_vector
        
    def _calculate_attention_entropy(self, attention_weights: np.ndarray) -> float:
        """Calculate entropy of attention distribution"""
        # Add small epsilon to avoid log(0)
        weights = attention_weights + 1e-8
        weights = weights / np.sum(weights)
        return -np.sum(weights * np.log(weights))
        
    def _extract_action(self, network_result: Dict) -> int:
        """Extract action from network activity"""
        # Simple action extraction based on module with highest activity
        activities = [network_result['activities'].get(i, 0.0) 
                     for i in range(len(self.network.modules))]
        return int(np.argmax(activities))
        
    def _reinforce_recent_activity(self, strength_factor: float):
        """Modulate strength of recent synaptic changes"""
        # Simple reinforcement: adjust learning rates temporarily
        for module in self.network.modules.values():
            for neuron in module.neurons:
                # Temporarily adjust adaptation (simplified)
                neuron.adaptation *= strength_factor
                
    def _memory_consolidation(self, steps: int):
        """Perform memory consolidation (replay of important patterns)"""
        if not self.learned_patterns:
            return
            
        # Replay important patterns
        important_patterns = list(self.learned_patterns.values())[-10:]  # Last 10 patterns
        
        for _ in range(steps):
            if important_patterns:
                pattern_data = np.random.choice(important_patterns)
                self.network.step(pattern_data['input'])
                
    def _test_task_performance(self, test_data: List[Tuple]) -> float:
        """Test performance on a task"""
        if not test_data:
            return 0.0
            
        correct = 0
        for input_pattern, expected_output in test_data:
            result = self.network.step(input_pattern)
            activity_vector = np.array([result['activities'].get(j, 0.0) 
                                      for j in range(len(self.network.modules))])
            target_vector = self._encode_target(expected_output)
            
            similarity = np.dot(activity_vector, target_vector) / (
                np.linalg.norm(activity_vector) * np.linalg.norm(target_vector) + 1e-8)
            
            if similarity > 0.5:
                correct += 1
                
        return correct / len(test_data)
        
    def visualize_training_progress(self, save_path: str = None):
        """Visualize training progress"""
        try:
            import matplotlib.pyplot as plt
            
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            
            # Plot accuracy if available
            if 'accuracy' in self.performance_metrics:
                axes[0, 0].plot(self.performance_metrics['accuracy'])
                axes[0, 0].set_title('Training Accuracy')
                axes[0, 0].set_xlabel('Epoch')
                axes[0, 0].set_ylabel('Accuracy')
                
            # Plot loss if available
            if 'loss' in self.performance_metrics:
                axes[0, 1].plot(self.performance_metrics['loss'])
                axes[0, 1].set_title('Training Loss')
                axes[0, 1].set_xlabel('Epoch')
                axes[0, 1].set_ylabel('Loss')
                
            # Plot activity if available
            if 'activity' in self.performance_metrics:
                axes[1, 0].plot(self.performance_metrics['activity'])
                axes[1, 0].set_title('Network Activity')
                axes[1, 0].set_xlabel('Step')
                axes[1, 0].set_ylabel('Activity Level')
                
            # Plot attention entropy if available
            if 'attention_entropy' in self.performance_metrics:
                axes[1, 1].plot(self.performance_metrics['attention_entropy'])
                axes[1, 1].set_title('Attention Entropy')
                axes[1, 1].set_xlabel('Step')
                axes[1, 1].set_ylabel('Entropy')
                
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                print(f"📊 Training progress saved to: {save_path}")
            else:
                plt.show()
                
        except ImportError:
            print("📊 Matplotlib not available - skipping visualization")
            
    def save_training_results(self, filepath: str):
        """Save training results to file"""
        results = {
            'performance_metrics': dict(self.performance_metrics),
            'learned_patterns': self.learned_patterns,
            'network_state': {
                'num_modules': len(self.network.modules),
                'attention_weights': self.network.attention_weights.tolist(),
                'working_memory_size': len(self.network.working_memory),
                'time_step': self.network.time_step
            },
            'timestamp': time.time()
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, default=str)
            
        print(f"💾 Training results saved to: {filepath}")


def create_sample_training_data():
    """Create sample training data for demonstration"""
    
    # Pattern recognition task
    pattern_data = []
    
    # Simple patterns
    patterns = {
        "vertical": {0: {i: True for i in range(0, 10, 2)}},
        "horizontal": {1: {i: True for i in range(5, 15)}},
        "diagonal": {2: {i: True for i in range(0, 15, 3)}},
        "cluster": {0: {i: True for i in [2, 3, 4, 12, 13, 14]}}
    }
    
    for pattern_name, pattern_dict in patterns.items():
        pattern_data.append((pattern_dict, pattern_name))
        
    return pattern_data


def demo_comprehensive_training():
    """Demonstrate comprehensive training approaches"""
    
    print("🎓 COMPREHENSIVE BRAIN NETWORK TRAINING DEMONSTRATION")
    print("=" * 70)
    
    # Create network and trainer
    network = SimpleBrainNetwork([40, 30, 25, 20])
    trainer = BrainNetworkTrainer(network)
    
    # Create sample data
    training_data = create_sample_training_data()
    
    print(f"\n📊 Training with {len(training_data)} pattern examples")
    
    # 1. Supervised Training
    print("\n🎯 Phase 1: Supervised Training")
    supervised_metrics = trainer.supervised_training(training_data, epochs=8)
    
    # 2. Unsupervised Training
    print("\n🧠 Phase 2: Unsupervised Training")
    input_patterns = [data[0] for data in training_data]
    unsupervised_metrics = trainer.unsupervised_training(input_patterns, training_steps=500)
    
    # 3. Simple Reinforcement Learning Environment
    print("\n🎮 Phase 3: Reinforcement Learning")
    
    def simple_environment(action, reset=False):
        """Simple environment that rewards certain actions"""
        if reset:
            return {0: {i: True for i in range(5)}}  # Initial state
            
        if action is None:
            return 0.0, {}, True
            
        # Reward based on action choice
        if action == 1:  # Prefer action 1
            reward = 1.0
        elif action == 0:  # Neutral
            reward = 0.0
        else:  # Negative for others
            reward = -0.5
            
        # Next state (random)
        next_state = {np.random.randint(0, 4): {i: True for i in range(np.random.randint(3, 8))}}
        done = np.random.random() < 0.1  # 10% chance of episode end
        
        return reward, next_state, done
        
    rl_metrics = trainer.reinforcement_training(simple_environment, num_episodes=50)
    
    # 4. Continual Learning
    print("\n🔄 Phase 4: Continual Learning")
    
    # Create sequence of tasks
    task1_data = training_data[:2]  # First 2 patterns
    task2_data = training_data[2:4]  # Last 2 patterns
    task3_data = [(training_data[0][0], "mixed"), (training_data[3][0], "mixed")]  # Mixed task
    
    task_sequence = [
        ("PatternTask1", task1_data),
        ("PatternTask2", task2_data),
        ("MixedTask", task3_data)
    ]
    
    continual_metrics = trainer.continual_learning(task_sequence)
    
    # Save results
    output_dir = "/Users/ravi.chillerega/sources/cde-hack-session/output"
    trainer.save_training_results(f"{output_dir}/comprehensive_training_results.json")
    
    # Visualize if possible
    try:
        trainer.visualize_training_progress(f"{output_dir}/training_progress.png")
    except Exception as e:
        print(f"Visualization failed: {e}")
        
    print("\n🎉 COMPREHENSIVE TRAINING COMPLETE!")
    print("=" * 70)
    
    # Summary
    print(f"\n📈 TRAINING SUMMARY:")
    print(f"   Supervised Accuracy: {supervised_metrics['accuracy'][-1]:.3f}")
    print(f"   Final Network Activity: {unsupervised_metrics['activity'][-1]:.3f}")
    print(f"   RL Average Reward: {np.mean(rl_metrics['episode_rewards'][-10:]):.3f}")
    print(f"   Patterns Learned: {len(trainer.learned_patterns)}")
    
    return trainer


if __name__ == "__main__":
    trainer = demo_comprehensive_training()
