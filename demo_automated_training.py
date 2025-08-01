#!/usr/bin/env python3
"""
Demonstration of HASN Automated Internet Training System
Shows how to use the complete automated training pipeline
"""

import asyncio
import time
import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, 'src')

async def demo_automated_training():
    """Demonstrate the automated training system"""
    
    print("🧠" + "=" * 60 + "🧠")
    print("  HASN AUTOMATED INTERNET TRAINING DEMO")
    print("  Live demonstration of self-learning AI")
    print("🧠" + "=" * 60 + "🧠")
    print()
    
    try:
        # Import training components
        from training.automated_internet_trainer import AutomatedInternetTrainer, TrainingConfig
        from training.training_monitor import TrainingMonitor
        
        print("✅ Training modules loaded successfully")
        
    except ImportError as e:
        print(f"❌ Error importing training modules: {e}")
        print("   Make sure you've run: python setup_automated_training.py")
        return
    
    # Create output directory
    Path("output").mkdir(exist_ok=True)
    
    print("\n🎯 Demo Configuration:")
    print("   • Articles per session: 8")
    print("   • Quality threshold: 0.4 (permissive for demo)")
    print("   • Request delay: 3 seconds (respectful)")
    print("   • Sources: Wikipedia + Reddit TIL")
    print()
    
    # Create demo configuration
    config = TrainingConfig(
        max_articles_per_session=8,  # Small for demo
        min_article_quality_score=0.4,  # Lower threshold for demo
        request_delay=3.0,  # Be respectful to servers
        max_concurrent_requests=2,  # Conservative
        sources=[
            'https://en.wikipedia.org/wiki/Special:Random',
            'https://www.reddit.com/r/todayilearned/top/.json?limit=10'
        ]
    )
    
    # Create trainer
    trainer = AutomatedInternetTrainer(config)
    
    print("🚀 Starting automated training demo...")
    print("   This will collect information from the internet")
    print("   and train the brain network in real-time.")
    print()
    
    # Show initial brain state
    initial_summary = trainer.get_knowledge_summary()
    print("🧠 Initial Brain State:")
    print(f"   Concepts known: {initial_summary['total_concepts']}")
    print(f"   Patterns stored: {initial_summary['trainer_patterns']}")
    print()
    
    # Start training (single session)
    start_time = time.time()
    print("📡 Collecting information from internet sources...")
    
    try:
        await trainer.start_training(continuous=False)
        
    except KeyboardInterrupt:
        print("\n🛑 Demo interrupted by user")
        return
    except Exception as e:
        print(f"\n❌ Training error: {e}")
        print("   This might be due to network issues or rate limiting")
        print("   The system is designed to handle these gracefully")
    
    duration = time.time() - start_time
    
    # Show results
    final_summary = trainer.get_knowledge_summary()
    
    print(f"\n✅ Training session completed in {duration:.1f} seconds")
    print("\n📊 Training Results:")
    print(f"   Concepts learned: {final_summary['total_concepts']}")
    print(f"   Patterns stored: {final_summary['trainer_patterns']}")
    
    # Show learned concepts
    if final_summary['concepts']:
        print("\n🧩 Recently Learned Concepts:")
        for concept in final_summary['concepts'][-10:]:
            print(f"      • {concept}")
    
    # Show brain state
    brain_state = final_summary.get('brain_state', {})
    if brain_state:
        print(f"\n🧠 Brain Network State:")
        print(f"   Working memory: {brain_state.get('working_memory', 'N/A')}")
        print(f"   Neural activity: {brain_state.get('total_activity', 'N/A')}")
    
    # Show training metrics
    training_metrics = final_summary.get('training_metrics', {})
    if training_metrics:
        print(f"\n📈 Training Metrics:")
        articles = training_metrics.get('articles_collected', 0)
        patterns = training_metrics.get('patterns_learned', 0)
        concepts = training_metrics.get('concepts_discovered', 0)
        
        print(f"   Articles processed: {articles}")
        print(f"   Neural patterns created: {patterns}")
        print(f"   New concepts discovered: {concepts}")
        
        if duration > 0:
            print(f"   Learning rate: {patterns/duration:.1f} patterns/second")
    
    # Generate monitoring report
    print("\n📋 Generating training report...")
    try:
        monitor = TrainingMonitor()
        report = monitor.generate_training_report()
        
        # Save report
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        report_file = f"output/demo_training_report_{timestamp}.txt"
        
        with open(report_file, 'w') as f:
            f.write(report)
        
        print(f"   Report saved to: {report_file}")
        
        # Show key metrics from report
        print("\n📊 Key Training Insights:")
        metrics_data = monitor.load_training_metrics()
        if metrics_data:
            latest = metrics_data[-1]
            latest_metrics = latest.get('training_metrics', {})
            
            quality_scores = latest_metrics.get('quality_scores', [])
            if quality_scores:
                import numpy as np
                avg_quality = np.mean(quality_scores)
                print(f"   Average content quality: {avg_quality:.3f}")
                print(f"   Quality range: {min(quality_scores):.3f} - {max(quality_scores):.3f}")
            
            # Show effectiveness of different concepts
            effectiveness = latest.get('pattern_effectiveness', {})
            if effectiveness:
                print("   Most effective learning topics:")
                sorted_topics = sorted(effectiveness.items(), 
                                     key=lambda x: x[1].get('effectiveness', 0), 
                                     reverse=True)
                for topic, info in sorted_topics[:3]:
                    eff_score = info.get('effectiveness', 0)
                    print(f"      • {topic}: {eff_score:.3f}")
        
    except Exception as e:
        print(f"   Report generation error: {e}")
    
    print("\n🎉 Demo Complete!")
    print("\nNext Steps:")
    print("=" * 20)
    print("1. Run continuous training:")
    print("   python src/training/train_cli.py start --continuous")
    print()
    print("2. Monitor training progress:")
    print("   python src/training/train_cli.py monitor")
    print()
    print("3. Start the API server:")
    print("   python src/api/main.py")
    print("   Then visit: http://localhost:8000/docs")
    print()
    print("4. View detailed documentation:")
    print("   cat src/training/AUTOMATED_TRAINING_README.md")
    print()
    print("🧠 Your HASN brain is now smarter and ready for continuous learning!")

def demo_brain_comparison():
    """Show before/after brain comparison"""
    print("\n🔬 Brain Analysis Demo")
    print("=" * 30)
    
    try:
        from core.advanced_brain_network import AdvancedCognitiveBrain
        
        # Create two brains - before and after learning
        print("Creating baseline brain...")
        baseline_brain = AdvancedCognitiveBrain()
        
        # Simulate some processing
        test_pattern = {
            0: {1: True, 5: True, 12: True},  # Sensory
            1: {3: True, 8: True, 15: True},  # Memory  
            2: {2: True, 7: True, 11: True},  # Executive
            3: {4: True, 9: True, 14: True}   # Motor
        }
        
        print("Processing test pattern through brain...")
        baseline_result = baseline_brain.process_pattern(test_pattern)
        
        print(f"Brain response:")
        print(f"   Total activity: {baseline_result.get('total_activity', 0):.3f}")
        print(f"   Working memory: {baseline_result.get('working_memory_size', 0)}/7")
        print(f"   Attention distribution: {baseline_result.get('attention_weights', [])}")
        
        print("\n💡 This demonstrates the brain's cognitive processing capabilities")
        print("   that will be enhanced through automated internet training!")
        
    except Exception as e:
        print(f"   Brain analysis error: {e}")

async def main():
    """Main demo function"""
    
    # Check setup
    print("🔍 Checking system setup...")
    
    required_files = [
        "src/training/automated_internet_trainer.py",
        "src/training/training_config.json",
        "src/core/brain_inspired_network.py"
    ]
    
    missing_files = []
    for file_path in required_files:
        if not os.path.exists(file_path):
            missing_files.append(file_path)
    
    if missing_files:
        print("❌ Missing required files:")
        for file_path in missing_files:
            print(f"   • {file_path}")
        print("\nPlease run: python setup_automated_training.py")
        return
    
    print("✅ System setup verified")
    
    # Ask user for demo type
    print("\n🎯 Demo Options:")
    print("1. Full automated training demo (recommended)")
    print("2. Brain analysis demo only") 
    print("3. Both demos")
    
    try:
        choice = input("\nEnter choice (1-3, default=1): ").strip() or "1"
        
        if choice in ["1", "3"]:
            await demo_automated_training()
        
        if choice in ["2", "3"]:
            demo_brain_comparison()
            
    except KeyboardInterrupt:
        print("\n👋 Demo cancelled by user")
    except Exception as e:
        print(f"\n❌ Demo error: {e}")

if __name__ == "__main__":
    asyncio.run(main())