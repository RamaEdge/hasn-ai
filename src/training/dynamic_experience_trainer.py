#!/usr/bin/env python3
"""
Dynamic Experience Trainer - Provides continuous learning experiences to brain networks
Replaces static experiences with dynamic, adaptive training scenarios
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from core.cognitive_brain_network import CognitiveBrainNetwork, CognitiveConfig
from core.simplified_brain_network import SimpleBrainNetwork, NetworkConfig, NetworkPresets
import numpy as np
import time
import random
from typing import Dict, List, Tuple, Optional, Generator
from dataclasses import dataclass
from collections import deque
import json


@dataclass
class ExperienceTemplate:
    """Template for generating dynamic experiences"""
    category: str
    pattern_generator: callable
    context_generator: callable
    difficulty_level: float
    learning_objectives: List[str]


class DynamicExperienceGenerator:
    """Generates dynamic, adaptive learning experiences"""
    
    def __init__(self):
        self.experience_templates = self._create_experience_templates()
        self.difficulty_progression = 0.1  # Start easy
        self.learned_concepts = set()
        self.experience_history = deque(maxlen=1000)
        
    def _create_experience_templates(self) -> Dict[str, ExperienceTemplate]:
        """Create templates for different types of experiences"""
        templates = {}
        
        # Mathematical relationships
        templates['arithmetic'] = ExperienceTemplate(
            category='arithmetic',
            pattern_generator=self._generate_arithmetic_pattern,
            context_generator=self._generate_arithmetic_context,
            difficulty_level=0.2,
            learning_objectives=['addition', 'subtraction', 'multiplication']
        )
        
        # Sequence learning
        templates['sequences'] = ExperienceTemplate(
            category='sequences',
            pattern_generator=self._generate_sequence_pattern,
            context_generator=self._generate_sequence_context,
            difficulty_level=0.3,
            learning_objectives=['pattern_recognition', 'prediction', 'temporal_learning']
        )
        
        # Categorization
        templates['categorization'] = ExperienceTemplate(
            category='categorization',
            pattern_generator=self._generate_category_pattern,
            context_generator=self._generate_category_context,
            difficulty_level=0.4,
            learning_objectives=['classification', 'abstraction', 'generalization']
        )
        
        # Analogical reasoning
        templates['analogies'] = ExperienceTemplate(
            category='analogies',
            pattern_generator=self._generate_analogy_pattern,
            context_generator=self._generate_analogy_context,
            difficulty_level=0.6,
            learning_objectives=['analogical_reasoning', 'transfer_learning', 'abstraction']
        )
        
        # Problem solving
        templates['problems'] = ExperienceTemplate(
            category='problems',
            pattern_generator=self._generate_problem_pattern,
            context_generator=self._generate_problem_context,
            difficulty_level=0.8,
            learning_objectives=['problem_solving', 'planning', 'multi_step_reasoning']
        )
        
        return templates
    
    def _generate_arithmetic_pattern(self, difficulty: float) -> Dict[int, bool]:
        """Generate arithmetic problem patterns"""
        # Scale numbers based on difficulty
        max_num = int(10 + difficulty * 90)  # 10-100 range
        a = random.randint(1, max_num)
        b = random.randint(1, max_num)
        
        # Encode numbers as neural patterns
        pattern = {}
        # Encode first number
        for i in range(min(a, 20)):  # Cap at 20 neurons
            pattern[i] = True
        # Encode second number with offset
        for i in range(min(b, 20)):
            pattern[30 + i] = True
        
        return pattern
    
    def _generate_arithmetic_context(self, difficulty: float) -> Dict:
        """Generate context for arithmetic problems"""
        operations = ['add', 'subtract', 'multiply']
        if difficulty < 0.3:
            operations = ['add', 'subtract']  # Easier operations
        
        return {
            'category': 'arithmetic',
            'operation': random.choice(operations),
            'difficulty': difficulty,
            'learning_objective': 'mathematical_reasoning'
        }
    
    def _generate_sequence_pattern(self, difficulty: float) -> Dict[int, bool]:
        """Generate sequence patterns"""
        # Create sequences of increasing complexity
        seq_length = int(3 + difficulty * 7)  # 3-10 length
        pattern = {}
        
        # Generate arithmetic or geometric sequence
        if random.random() < 0.5:
            # Arithmetic sequence
            start = random.randint(1, 5)
            step = random.randint(1, 3)
            sequence = [start + i * step for i in range(seq_length)]
        else:
            # Pattern sequence (e.g., alternating, doubling)
            base_pattern = [1, 2, 1, 2] if difficulty < 0.5 else [1, 2, 4, 8]
            sequence = (base_pattern * (seq_length // len(base_pattern) + 1))[:seq_length]
        
        # Encode sequence as neural pattern
        for i, val in enumerate(sequence):
            for j in range(min(val, 5)):  # Encode value as multiple active neurons
                pattern[i * 10 + j] = True
        
        return pattern
    
    def _generate_sequence_context(self, difficulty: float) -> Dict:
        """Generate context for sequence learning"""
        sequence_types = ['arithmetic', 'geometric', 'alternating', 'fibonacci']
        
        return {
            'category': 'sequence',
            'sequence_type': random.choice(sequence_types),
            'difficulty': difficulty,
            'learning_objective': 'pattern_prediction',
            'requires_prediction': True
        }
    
    def _generate_category_pattern(self, difficulty: float) -> Dict[int, bool]:
        """Generate categorization patterns"""
        categories = {
            'animals': [1, 2, 3, 4, 5],
            'vehicles': [10, 11, 12, 13, 14],
            'foods': [20, 21, 22, 23, 24],
            'tools': [30, 31, 32, 33, 34]
        }
        
        # Select category and add noise based on difficulty
        category = random.choice(list(categories.keys()))
        base_pattern = categories[category]
        
        pattern = {}
        for neuron in base_pattern:
            if random.random() > difficulty * 0.3:  # Less noise for easier problems
                pattern[neuron] = True
        
        # Add some random noise
        max_noise = max(1, int(difficulty * 5))
        noise_neurons = random.randint(1, max_noise)
        for _ in range(noise_neurons):
            pattern[random.randint(50, 99)] = True
        
        return pattern
    
    def _generate_category_context(self, difficulty: float) -> Dict:
        """Generate context for categorization"""
        categories = ['animals', 'vehicles', 'foods', 'tools']
        
        return {
            'category': 'classification',
            'target_category': random.choice(categories),
            'difficulty': difficulty,
            'learning_objective': 'categorization',
            'requires_classification': True
        }
    
    def _generate_analogy_pattern(self, difficulty: float) -> Dict[int, bool]:
        """Generate analogical reasoning patterns"""
        # A:B :: C:? type patterns
        analogies = {
            'size': ([1, 2], [1, 2, 3, 4, 5]),  # small -> large
            'quantity': ([1], [1, 1, 1]),        # one -> many
            'opposite': ([1, 2], [8, 9]),        # left -> right (encoded differently)
        }
        
        analogy_type = random.choice(list(analogies.keys()))
        pattern_a, pattern_b = analogies[analogy_type]
        
        pattern = {}
        # Encode A term
        for neuron in pattern_a:
            pattern[neuron] = True
        # Encode B term
        for neuron in pattern_b:
            pattern[10 + neuron] = True
        # Encode C term (similar to A)
        for neuron in pattern_a:
            pattern[20 + neuron] = True
        
        return pattern
    
    def _generate_analogy_context(self, difficulty: float) -> Dict:
        """Generate context for analogical reasoning"""
        analogy_types = ['size', 'quantity', 'opposite', 'function', 'category']
        
        return {
            'category': 'analogy',
            'analogy_type': random.choice(analogy_types),
            'difficulty': difficulty,
            'learning_objective': 'analogical_reasoning',
            'requires_inference': True
        }
    
    def _generate_problem_pattern(self, difficulty: float) -> Dict[int, bool]:
        """Generate problem-solving patterns"""
        # Multi-step problems requiring planning
        problem_size = int(2 + difficulty * 6)  # 2-8 steps
        
        pattern = {}
        # Encode problem state
        for i in range(problem_size):
            # Each step encoded as a sub-pattern
            base_neuron = i * 10
            step_complexity = random.randint(1, int(1 + difficulty * 4))
            for j in range(step_complexity):
                pattern[base_neuron + j] = True
        
        return pattern
    
    def _generate_problem_context(self, difficulty: float) -> Dict:
        """Generate context for problem solving"""
        problem_types = ['planning', 'optimization', 'constraint_satisfaction', 'search']
        
        return {
            'category': 'problem_solving',
            'problem_type': random.choice(problem_types),
            'difficulty': difficulty,
            'learning_objective': 'multi_step_reasoning',
            'requires_planning': True
        }
    
    def generate_experience(self, target_difficulty: Optional[float] = None) -> Tuple[Dict[int, bool], Dict]:
        """Generate a dynamic learning experience"""
        # Select difficulty
        if target_difficulty is None:
            target_difficulty = self.difficulty_progression
        
        # Select appropriate template based on difficulty and learning progress
        suitable_templates = [
            name for name, template in self.experience_templates.items()
            if abs(template.difficulty_level - target_difficulty) < 0.3
        ]
        
        if not suitable_templates:
            suitable_templates = list(self.experience_templates.keys())
        
        template_name = random.choice(suitable_templates)
        template = self.experience_templates[template_name]
        
        # Generate experience
        pattern = template.pattern_generator(target_difficulty)
        context = template.context_generator(target_difficulty)
        
        # Add metadata
        context['experience_id'] = f"exp_{int(time.time() * 1000)}_{len(self.experience_history)}"
        context['template'] = template_name
        context['generated_at'] = time.time()
        
        # Record experience
        self.experience_history.append({
            'experience_id': context['experience_id'],
            'template': template_name,
            'difficulty': target_difficulty,
            'timestamp': time.time()
        })
        
        return pattern, context
    
    def adapt_difficulty(self, success_rate: float):
        """Adapt difficulty based on learning performance"""
        if success_rate > 0.8:
            self.difficulty_progression = min(1.0, self.difficulty_progression + 0.05)
        elif success_rate < 0.4:
            self.difficulty_progression = max(0.1, self.difficulty_progression - 0.05)
        
        print(f"🎯 Adapted difficulty to {self.difficulty_progression:.2f} (success rate: {success_rate:.2f})")
    
    def get_curriculum(self, num_experiences: int) -> List[Tuple[Dict[int, bool], Dict]]:
        """Generate a curriculum of progressive experiences"""
        curriculum = []
        
        for i in range(num_experiences):
            # Progressive difficulty
            difficulty = 0.1 + (i / num_experiences) * 0.8
            pattern, context = self.generate_experience(difficulty)
            curriculum.append((pattern, context))
        
        return curriculum


class AdaptiveTrainingSession:
    """Manages adaptive training sessions with dynamic experiences"""
    
    def __init__(self, network: CognitiveBrainNetwork):
        self.network = network
        self.experience_generator = DynamicExperienceGenerator()
        self.training_metrics = {
            'experiences_processed': 0,
            'successful_inferences': 0,
            'memories_formed': 0,
            'associations_created': 0,
            'session_start': time.time()
        }
        self.performance_history = deque(maxlen=100)
    
    def run_adaptive_session(self, duration_minutes: int = 10, experiences_per_minute: int = 6):
        """Run an adaptive training session with dynamic experiences"""
        print(f"🚀 Starting adaptive training session ({duration_minutes} minutes)")
        print(f"   Target: {experiences_per_minute} experiences per minute")
        
        session_start = time.time()
        session_end = session_start + (duration_minutes * 60)
        experience_interval = 60.0 / experiences_per_minute
        
        last_experience_time = session_start
        
        while time.time() < session_end:
            current_time = time.time()
            
            # Generate experience at regular intervals
            if current_time - last_experience_time >= experience_interval:
                self._process_single_experience()
                last_experience_time = current_time
                
                # Adapt difficulty every 10 experiences
                if self.training_metrics['experiences_processed'] % 10 == 0:
                    self._adapt_training_difficulty()
            
            # Brief pause to prevent overwhelming
            time.sleep(0.1)
        
        self._print_session_summary()
    
    def _process_single_experience(self):
        """Process a single dynamic experience"""
        # Generate experience
        pattern, context = self.experience_generator.generate_experience()
        
        # Add inference request for cognitive evaluation
        context['generate_inferences'] = True
        
        # Process through network
        result = self.network.step_with_cognition(pattern, context)
        
        # Evaluate performance
        performance = self._evaluate_performance(result, context)
        self.performance_history.append(performance)
        
        # Update metrics
        self.training_metrics['experiences_processed'] += 1
        if result.get('memory_id'):
            self.training_metrics['memories_formed'] += 1
        if result.get('inferences') and len(result['inferences']) > 0:
            self.training_metrics['successful_inferences'] += 1
        
        # Print progress occasionally
        if self.training_metrics['experiences_processed'] % 5 == 0:
            self._print_progress()
    
    def _evaluate_performance(self, result: Dict, context: Dict) -> float:
        """Evaluate the quality of the network's response"""
        performance_score = 0.0
        
        # Memory formation (basic requirement)
        if result.get('memory_id'):
            performance_score += 0.3
        
        # Spike activity (indicates processing)
        spike_count = result.get('spike_count', 0)
        if spike_count > 0:
            performance_score += 0.2
        
        # Inference generation (higher-level cognition)
        inferences = result.get('inferences', [])
        if inferences:
            performance_score += 0.3
            # Bonus for high-confidence inferences
            avg_confidence = sum(inf['confidence'] for inf in inferences) / len(inferences)
            performance_score += 0.2 * avg_confidence
        
        return performance_score
    
    def _adapt_training_difficulty(self):
        """Adapt training difficulty based on recent performance"""
        if len(self.performance_history) >= 10:
            recent_performance = list(self.performance_history)[-10:]
            success_rate = sum(p > 0.5 for p in recent_performance) / len(recent_performance)
            self.experience_generator.adapt_difficulty(success_rate)
    
    def _print_progress(self):
        """Print training progress"""
        exp_count = self.training_metrics['experiences_processed']
        mem_count = self.training_metrics['memories_formed']
        inf_count = self.training_metrics['successful_inferences']
        
        recent_perf = np.mean(list(self.performance_history)[-5:]) if self.performance_history else 0
        
        print(f"   📊 Experience {exp_count}: {mem_count} memories, {inf_count} inferences, performance: {recent_perf:.2f}")
    
    def _print_session_summary(self):
        """Print comprehensive session summary"""
        duration = time.time() - self.training_metrics['session_start']
        cognitive_state = self.network.get_cognitive_state()
        
        print(f"\n🎉 Training Session Complete!")
        print(f"=" * 50)
        print(f"Duration: {duration/60:.1f} minutes")
        print(f"Experiences processed: {self.training_metrics['experiences_processed']}")
        print(f"Memories formed: {self.training_metrics['memories_formed']}")
        print(f"Successful inferences: {self.training_metrics['successful_inferences']}")
        print(f"Final difficulty level: {self.experience_generator.difficulty_progression:.2f}")
        
        if self.performance_history:
            avg_performance = np.mean(list(self.performance_history))
            print(f"Average performance: {avg_performance:.2f}")
        
        print(f"\n🧠 Cognitive State:")
        for key, value in cognitive_state.items():
            if isinstance(value, float):
                print(f"   {key}: {value:.3f}")
            else:
                print(f"   {key}: {value}")


def demo_dynamic_training():
    """Demonstrate dynamic experience training"""
    print("🧠 Dynamic Experience Training Demo")
    print("=" * 50)
    
    # Create cognitive network
    config = CognitiveConfig(
        learning_rate=0.04,
        learning_probability=0.6,
        max_episodic_memories=200,
        association_strength_threshold=0.2,
        max_inference_depth=3
    )
    
    network = CognitiveBrainNetwork(num_neurons=150, config=config)
    
    # Create adaptive training session
    trainer = AdaptiveTrainingSession(network)
    
    # Run training session
    trainer.run_adaptive_session(duration_minutes=2, experiences_per_minute=10)
    
    print(f"\n🎯 Testing learned capabilities...")
    
    # Test inference on new experience
    test_pattern = {1: True, 2: True, 3: True, 15: True, 16: True, 17: True}
    test_context = {
        'category': 'test',
        'type': 'inference_test',
        'generate_inferences': True
    }
    
    result = network.step_with_cognition(test_pattern, test_context)
    
    print(f"Test inferences generated: {len(result.get('inferences', []))}")
    for i, inference in enumerate(result.get('inferences', [])[:3]):
        print(f"   {i+1}. Confidence: {inference['confidence']:.3f}")
    
    return network


if __name__ == "__main__":
    demo_dynamic_training()