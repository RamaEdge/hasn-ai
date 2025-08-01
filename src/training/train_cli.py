#!/usr/bin/env python3
"""
Command Line Interface for HASN Automated Internet Training
Easy-to-use CLI for starting, stopping, and managing automated training
"""

import asyncio
import argparse
import json
import os
import sys
import signal
from datetime import datetime
from typing import Optional

# Add path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from training.automated_internet_trainer import AutomatedInternetTrainer, TrainingConfig
    from training.training_monitor import TrainingMonitor
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Make sure you're running from the correct directory and all dependencies are installed.")
    sys.exit(1)

class TrainingCLI:
    """Command-line interface for automated training"""
    
    def __init__(self):
        self.trainer: Optional[AutomatedInternetTrainer] = None
        self.training_task: Optional[asyncio.Task] = None
        self.monitor = TrainingMonitor()
        
    def load_config(self, profile: str = "development") -> TrainingConfig:
        """Load training configuration from file"""
        config_file = os.path.join(os.path.dirname(__file__), 'training_config.json')
        
        try:
            with open(config_file, 'r') as f:
                config_data = json.load(f)
            
            profile_config = config_data['training_profiles'].get(profile)
            if not profile_config:
                print(f"⚠️  Profile '{profile}' not found. Using development profile.")
                profile_config = config_data['training_profiles']['development']
            
            # Create TrainingConfig object
            config = TrainingConfig(
                max_articles_per_session=profile_config['max_articles_per_session'],
                collection_interval=profile_config['collection_interval'],
                min_article_quality_score=profile_config['min_article_quality_score'],
                learning_rate=profile_config['learning_rate'],
                save_interval=profile_config['save_interval'],
                max_concurrent_requests=profile_config['max_concurrent_requests'],
                request_delay=profile_config['request_delay'],
                sources=profile_config['sources'],
                content_filters=profile_config['content_filters']
            )
            
            print(f"✅ Loaded '{profile}' configuration profile")
            return config
            
        except FileNotFoundError:
            print(f"⚠️  Config file not found: {config_file}")
            print("Using default configuration...")
            return TrainingConfig()
        except Exception as e:
            print(f"❌ Error loading config: {e}")
            print("Using default configuration...")
            return TrainingConfig()
    
    async def start_training(self, profile: str, continuous: bool, load_state: Optional[str]):
        """Start automated training"""
        print("🧠 HASN Automated Internet Training")
        print("=" * 50)
        
        # Load configuration
        config = self.load_config(profile)
        
        # Create trainer
        self.trainer = AutomatedInternetTrainer(config)
        
        # Load previous state if specified
        if load_state:
            if os.path.exists(load_state):
                await self.trainer.load_training_state(load_state)
                print(f"📂 Loaded previous training state from: {load_state}")
            else:
                print(f"⚠️  State file not found: {load_state}")
        
        # Display configuration
        print(f"\n⚙️  Training Configuration:")
        print(f"   Profile: {profile}")
        print(f"   Max Articles per Session: {config.max_articles_per_session}")
        print(f"   Collection Interval: {config.collection_interval}s")
        print(f"   Min Quality Score: {config.min_article_quality_score}")
        print(f"   Sources: {len(config.sources)} configured")
        print(f"   Continuous Mode: {'Yes' if continuous else 'No'}")
        print()
        
        # Set up signal handler for graceful shutdown
        def signal_handler(signum, frame):
            print("\n🛑 Received interrupt signal. Stopping training...")
            if self.training_task:
                self.training_task.cancel()
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        
        try:
            # Start training
            print("🚀 Starting automated training...")
            if continuous:
                print("   Running in continuous mode. Press Ctrl+C to stop and save.")
            else:
                print("   Running single session.")
            print()
            
            await self.trainer.start_training(continuous=continuous)
            
        except asyncio.CancelledError:
            print("🛑 Training stopped by user")
        except Exception as e:
            print(f"❌ Training error: {e}")
        finally:
            if self.trainer:
                # Show final summary
                summary = self.trainer.get_knowledge_summary()
                print("\n📊 Training Session Summary:")
                print(f"   Total Concepts: {summary['total_concepts']}")
                print(f"   Total Patterns: {summary['trainer_patterns']}")
                if summary['concepts']:
                    print(f"   Recent Concepts: {', '.join(summary['concepts'][-5:])}")
                print()
                print("💾 Training state has been automatically saved.")
    
    def show_status(self):
        """Show current training status"""
        metrics_data = self.monitor.load_training_metrics()
        
        if not metrics_data:
            print("❌ No training data found.")
            print("   Start training first with: python train_cli.py start")
            return
        
        latest = metrics_data[-1]
        latest_metrics = latest.get('training_metrics', {})
        
        print("🧠 HASN Training Status")
        print("=" * 30)
        print(f"Total Sessions: {len(metrics_data)}")
        print(f"Articles Collected: {latest_metrics.get('articles_collected', 0)}")
        print(f"Patterns Learned: {latest_metrics.get('patterns_learned', 0)}")
        print(f"Concepts Discovered: {latest_metrics.get('concepts_discovered', 0)}")
        
        if latest_metrics.get('quality_scores'):
            import numpy as np
            avg_quality = np.mean(latest_metrics['quality_scores'])
            print(f"Average Quality: {avg_quality:.3f}")
        
        # Show recent concepts
        concepts = latest.get('learned_concepts', [])
        if concepts:
            print(f"Recent Concepts: {', '.join(concepts[-10:])}")
        
        # Show last update time
        timestamp = latest.get('timestamp', '')
        if timestamp:
            try:
                last_time = datetime.fromisoformat(timestamp.replace('_', 'T'))
                time_since = datetime.now() - last_time
                print(f"Last Update: {time_since} ago")
            except:
                print(f"Last Update: {timestamp}")
    
    def list_profiles(self):
        """List available training profiles"""
        config_file = os.path.join(os.path.dirname(__file__), 'training_config.json')
        
        try:
            with open(config_file, 'r') as f:
                config_data = json.load(f)
            
            profiles = config_data['training_profiles']
            
            print("📋 Available Training Profiles:")
            print("=" * 35)
            
            for name, config in profiles.items():
                print(f"\n🔧 {name.upper()}:")
                print(f"   Articles per session: {config['max_articles_per_session']}")
                print(f"   Collection interval: {config['collection_interval']}s")
                print(f"   Min quality score: {config['min_article_quality_score']}")
                print(f"   Sources: {len(config['sources'])}")
                
                # Show use case recommendation
                if name == 'development':
                    print("   💡 Best for: Testing and development")
                elif name == 'production':
                    print("   💡 Best for: Regular automated training")
                elif name == 'research':
                    print("   💡 Best for: Intensive research and learning")
            
        except Exception as e:
            print(f"❌ Error loading profiles: {e}")
    
    def cleanup_old_files(self, days: int = 7):
        """Clean up old training files"""
        output_dir = "output"
        if not os.path.exists(output_dir):
            print("📁 No output directory found.")
            return
        
        import time
        from datetime import timedelta
        
        cutoff_time = time.time() - (days * 24 * 60 * 60)
        cleaned_count = 0
        
        for filename in os.listdir(output_dir):
            if filename.startswith('automated_') and filename.endswith('.json'):
                filepath = os.path.join(output_dir, filename)
                file_time = os.path.getmtime(filepath)
                
                if file_time < cutoff_time:
                    try:
                        os.remove(filepath)
                        cleaned_count += 1
                        print(f"🗑️  Removed: {filename}")
                    except Exception as e:
                        print(f"⚠️  Error removing {filename}: {e}")
        
        print(f"\n✅ Cleanup complete. Removed {cleaned_count} old files.")

async def main():
    """Main CLI function"""
    parser = argparse.ArgumentParser(
        description='HASN Automated Internet Training CLI',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s start                          # Start training with development profile
  %(prog)s start --profile production     # Start with production profile
  %(prog)s start --continuous             # Run continuously until stopped
  %(prog)s start --load-state output/state.json  # Resume from saved state
  %(prog)s status                         # Show current training status
  %(prog)s report                         # Generate detailed report
  %(prog)s monitor                        # Real-time monitoring
  %(prog)s profiles                       # List available profiles
  %(prog)s cleanup --days 7               # Clean up files older than 7 days
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Start command
    start_parser = subparsers.add_parser('start', help='Start automated training')
    start_parser.add_argument('--profile', default='development', 
                             choices=['development', 'production', 'research'],
                             help='Training profile to use')
    start_parser.add_argument('--continuous', action='store_true',
                             help='Run continuously until stopped')
    start_parser.add_argument('--load-state', help='Load previous training state from file')
    
    # Status command
    subparsers.add_parser('status', help='Show current training status')
    
    # Report command
    report_parser = subparsers.add_parser('report', help='Generate training report')
    report_parser.add_argument('--save', help='Save report to file')
    
    # Monitor command
    monitor_parser = subparsers.add_parser('monitor', help='Real-time training monitor')
    monitor_parser.add_argument('--refresh', type=int, default=30,
                               help='Refresh interval in seconds')
    
    # Visualize command
    viz_parser = subparsers.add_parser('visualize', help='Create training visualization')
    viz_parser.add_argument('--save', help='Save visualization to file')
    
    # Profiles command
    subparsers.add_parser('profiles', help='List available training profiles')
    
    # Cleanup command
    cleanup_parser = subparsers.add_parser('cleanup', help='Clean up old training files')
    cleanup_parser.add_argument('--days', type=int, default=7,
                               help='Remove files older than N days')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    cli = TrainingCLI()
    
    try:
        if args.command == 'start':
            await cli.start_training(args.profile, args.continuous, args.load_state)
        
        elif args.command == 'status':
            cli.show_status()
        
        elif args.command == 'report':
            report = cli.monitor.generate_training_report()
            print(report)
            
            if args.save:
                with open(args.save, 'w') as f:
                    f.write(report)
                print(f"\n💾 Report saved to: {args.save}")
        
        elif args.command == 'monitor':
            cli.monitor.real_time_monitor(args.refresh)
        
        elif args.command == 'visualize':
            cli.monitor.create_visualization(args.save)
        
        elif args.command == 'profiles':
            cli.list_profiles()
        
        elif args.command == 'cleanup':
            cli.cleanup_old_files(args.days)
    
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())