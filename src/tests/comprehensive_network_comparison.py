"""
Comprehensive comparison of all three brain network implementations:
1. SimpleBrainNetwork (simplified_brain_network.py)
2. AdvancedBrainInspiredNetwork (advanced_brain_network.py) 
3. OptimizedHASN (optimized_brain_network.py)

This test will determine which network provides the best performance
and functionality balance to guide our consolidation decision.
"""

import time
import numpy as np
import sys
import os
import tracemalloc
import psutil
from typing import Dict, List, Tuple, Any

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# Import all three implementations
from core.simplified_brain_network import SimpleBrainNetwork
from core.cognitive_brain_network import CognitiveBrainNetwork, CognitiveConfig


class NetworkComparator:
    """Compare performance and functionality of different brain networks"""
    
    def __init__(self):
        self.results = {}
        
    def measure_memory_usage(self, func, *args, **kwargs):
        """Measure memory usage of a function"""
        tracemalloc.start()
        result = func(*args, **kwargs)
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        
        return result, {
            'current_mb': current / 1024 / 1024,
            'peak_mb': peak / 1024 / 1024
        }
    
    def create_equivalent_networks(self, num_neurons: int = 100) -> Dict[str, Any]:
        """Create equivalent networks for comparison"""
        networks = {}
        
        print(f"Creating networks with ~{num_neurons} neurons each...")
        
        # 1. Simplified Brain Network
        print("  Creating SimpleBrainNetwork...")
        try:
            start_time = time.time()
            networks['simplified'] = SimpleBrainNetwork(
                num_neurons=num_neurons,
                connectivity_prob=0.1
            )
            networks['simplified_creation_time'] = time.time() - start_time
            print(f"    ✅ Created in {networks['simplified_creation_time']:.3f}s")
        except Exception as e:
            print(f"    ❌ Failed: {e}")
            networks['simplified'] = None
        
        # 2. Cognitive Brain Network
        print("  Creating CognitiveBrainNetwork...")
        try:
            start_time = time.time()
            networks['advanced'] = CognitiveBrainNetwork(num_neurons=num_neurons, connectivity_prob=0.1, config=CognitiveConfig())
            networks['advanced_creation_time'] = time.time() - start_time
            print(f"    ✅ Created in {networks['advanced_creation_time']:.3f}s")
        except Exception as e:
            print(f"    ❌ Failed: {e}")
            networks['advanced'] = None
        
        # 3. Optimized network not present in codebase
        networks['optimized'] = None
        
        return networks
    
    def run_performance_test(self, network, network_name: str, duration: int = 1000) -> Dict[str, Any]:
        """Run performance test on a network"""
        print(f"  Testing {network_name}...")
        
        results = {
            'network_name': network_name,
            'duration': duration,
            'success': False
        }
        
        if network is None:
            results['error'] = 'Network creation failed'
            return results
        
        try:
            # Create input pattern
            def create_input(t):
                if network_name == 'simplified':
                    # Simple input for SimpleBrainNetwork
                    return {i: np.random.random() < 0.1 for i in range(5)}
                elif network_name == 'advanced':
                    # Input for CognitiveBrainNetwork: external_input bools
                    return {i: (np.random.random() < 0.1) for i in range(10)}
                elif network_name == 'optimized':
                    # Input for OptimizedHASN
                    return {0: {i: np.random.random() < 0.1 for i in range(5)}}
                return {}
            
            # Measure execution time and memory
            start_time = time.time()
            tracemalloc.start()
            
            spikes_recorded = 0
            
            # Run simulation
            if network_name == 'simplified':
                sim_results = network.run_simulation(
                    duration=duration,
                    input_pattern_func=create_input
                )
                spikes_recorded = len(sim_results.get('spike_record', []))
                
            elif network_name == 'advanced':
                # Run step-by-step for cognitive network
                for step in range(duration):
                    input_data = create_input(step)
                    result = network.step_with_cognition(input_data, context={})
                    spikes_recorded += result.get('spike_count', 0)
                
            elif network_name == 'optimized':
                pass
            
            # Measure final metrics
            execution_time = time.time() - start_time
            current_memory, peak_memory = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            
            # Calculate performance metrics
            steps_per_second = duration / execution_time if execution_time > 0 else 0
            memory_mb = peak_memory / 1024 / 1024
            
            results.update({
                'success': True,
                'execution_time': execution_time,
                'steps_per_second': steps_per_second,
                'memory_mb': memory_mb,
                'spikes_recorded': spikes_recorded,
                'spikes_per_second': spikes_recorded / execution_time if execution_time > 0 else 0
            })
            
            print(f"    ✅ Completed in {execution_time:.3f}s")
            print(f"       Speed: {steps_per_second:.1f} steps/sec")
            print(f"       Memory: {memory_mb:.1f} MB")
            print(f"       Spikes: {spikes_recorded}")
            
        except Exception as e:
            results['error'] = str(e)
            print(f"    ❌ Failed: {e}")
            tracemalloc.stop()
        
        return results
    
    def analyze_functionality(self, networks: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        """Analyze functionality features of each network"""
        functionality = {}
        
        for network_name in ['simplified', 'advanced', 'optimized']:
            network = networks.get(network_name)
            
            if network is None:
                functionality[network_name] = {'available': False}
                continue
            
            features = {
                'available': True,
                'spiking_neurons': True,  # All have this
                'learning_mechanism': False,
                'memory_systems': False,
                'attention_mechanism': False,
                'cognitive_modules': False,
                'vectorized_operations': False,
                'sparse_connectivity': False,
                'real_time_capable': False
            }
            
            # Analyze specific features
            if network_name == 'simplified':
                features.update({
                    'learning_mechanism': True,  # Hebbian learning
                    'real_time_capable': True,   # Due to simplicity
                    'complexity_level': 'Low',
                    'primary_focus': 'Performance and Simplicity'
                })
                
            elif network_name == 'advanced':
                features.update({
                    'learning_mechanism': True,
                    'memory_systems': True,      # Working memory
                    'attention_mechanism': True,  # Attention system
                    'cognitive_modules': True,   # Specialized modules
                    'complexity_level': 'High',
                    'primary_focus': 'Cognitive Capabilities'
                })
                
            elif network_name == 'optimized':
                features.update({
                    'learning_mechanism': True,   # STDP
                    'vectorized_operations': True, # Numpy vectorization
                    'sparse_connectivity': True,  # Scipy sparse matrices
                    'real_time_capable': True,   # Optimized for performance
                    'complexity_level': 'Medium',
                    'primary_focus': 'Performance Optimization'
                })
            
            functionality[network_name] = features
        
        return functionality
    
    def comprehensive_comparison(self) -> Dict[str, Any]:
        """Run comprehensive comparison of all networks"""
        print("=" * 80)
        print("COMPREHENSIVE BRAIN NETWORK COMPARISON")
        print("=" * 80)
        
        # Create networks
        networks = self.create_equivalent_networks(100)
        
        # Test performance with different workloads
        performance_results = {}
        
        for test_name, duration in [('Quick', 100), ('Standard', 1000), ('Intensive', 5000)]:
            print(f"\n🧪 {test_name} Performance Test ({duration} steps):")
            performance_results[test_name.lower()] = {}
            
            for network_name in ['simplified', 'advanced', 'optimized']:
                network = networks.get(network_name)
                result = self.run_performance_test(network, network_name, duration)
                performance_results[test_name.lower()][network_name] = result
        
        # Analyze functionality
        print(f"\n🔍 Functionality Analysis:")
        functionality_analysis = self.analyze_functionality(networks)
        
        # Generate recommendations
        recommendations = self.generate_recommendations(performance_results, functionality_analysis)
        
        return {
            'networks_created': {k: v is not None for k, v in networks.items() if not k.endswith('_time')},
            'creation_times': {k: v for k, v in networks.items() if k.endswith('_time')},
            'performance_results': performance_results,
            'functionality_analysis': functionality_analysis,
            'recommendations': recommendations
        }
    
    def generate_recommendations(self, performance: Dict, functionality: Dict) -> Dict[str, Any]:
        """Generate recommendations based on test results"""
        print(f"\n📊 Generating Recommendations...")
        
        # Calculate average performance scores
        performance_scores = {}
        functionality_scores = {}
        
        for network_name in ['simplified', 'advanced', 'optimized']:
            # Performance score (higher is better)
            perf_data = performance.get('standard', {}).get(network_name, {})
            if perf_data.get('success', False):
                speed_score = perf_data.get('steps_per_second', 0)
                memory_score = 1000 / max(perf_data.get('memory_mb', 1), 1)  # Lower memory is better
                performance_scores[network_name] = speed_score + memory_score
            else:
                performance_scores[network_name] = 0
            
            # Functionality score
            func_data = functionality.get(network_name, {})
            if func_data.get('available', False):
                feature_count = sum(1 for k, v in func_data.items() 
                                   if isinstance(v, bool) and v and k != 'available')
                functionality_scores[network_name] = feature_count
            else:
                functionality_scores[network_name] = 0
        
        # Find best performers
        best_performance = max(performance_scores, key=performance_scores.get)
        best_functionality = max(functionality_scores, key=functionality_scores.get)
        
        # Calculate overall scores (balanced approach)
        overall_scores = {}
        for network_name in performance_scores:
            perf_norm = performance_scores[network_name] / max(performance_scores[network_name] for network_name in performance_scores) if max(performance_scores.values()) > 0 else 0
            func_norm = functionality_scores[network_name] / max(functionality_scores.values()) if max(functionality_scores.values()) > 0 else 0
            overall_scores[network_name] = perf_norm * 0.6 + func_norm * 0.4  # Weight performance higher
        
        best_overall = max(overall_scores, key=overall_scores.get)
        
        recommendations = {
            'best_performance': best_performance,
            'best_functionality': best_functionality,
            'best_overall': best_overall,
            'performance_scores': performance_scores,
            'functionality_scores': functionality_scores,
            'overall_scores': overall_scores,
            'recommendation': None
        }
        
        # Generate specific recommendation
        if best_overall == 'simplified':
            recommendations['recommendation'] = {
                'keep': 'simplified_brain_network.py',
                'remove': ['advanced_brain_network.py', 'optimized_brain_network.py'],
                'reason': 'Simplified network provides the best balance of performance and maintainability. It\'s fast, reliable, and covers all essential brain-inspired functionality.'
            }
        elif best_overall == 'optimized':
            recommendations['recommendation'] = {
                'keep': 'optimized_brain_network.py', 
                'remove': ['simplified_brain_network.py', 'advanced_brain_network.py'],
                'reason': 'Optimized network provides superior performance while maintaining good functionality. Best choice for production systems.'
            }
        elif best_overall == 'advanced':
            recommendations['recommendation'] = {
                'keep': 'advanced_brain_network.py',
                'remove': ['simplified_brain_network.py', 'optimized_brain_network.py'],
                'reason': 'Advanced network provides the most comprehensive cognitive capabilities. Best for research and complex applications.'
            }
        
        return recommendations


def main():
    """Run comprehensive network comparison"""
    comparator = NetworkComparator()
    results = comparator.comprehensive_comparison()
    
    print("\n" + "=" * 80)
    print("FINAL RECOMMENDATIONS")
    print("=" * 80)
    
    rec = results['recommendations']['recommendation']
    if rec:
        print(f"🎯 RECOMMENDED ACTION:")
        print(f"   KEEP: {rec['keep']}")
        print(f"   REMOVE: {', '.join(rec['remove'])}")
        print(f"   REASON: {rec['reason']}")
        
        print(f"\n📊 PERFORMANCE SCORES:")
        for network, score in results['recommendations']['performance_scores'].items():
            print(f"   {network}: {score:.1f}")
            
        print(f"\n🔧 FUNCTIONALITY SCORES:")
        for network, score in results['recommendations']['functionality_scores'].items():
            print(f"   {network}: {score}")
            
        print(f"\n🏆 OVERALL SCORES:")
        for network, score in results['recommendations']['overall_scores'].items():
            print(f"   {network}: {score:.3f}")
    
    return results


if __name__ == "__main__":
    results = main()