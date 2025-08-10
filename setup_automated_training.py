#!/usr/bin/env python3
"""
Setup script for HASN Automated Training System
Installs dependencies and prepares the system for first use
"""

import os
import sys
import subprocess
import json
from pathlib import Path

def print_banner():
    """Print setup banner"""
    print("🧠" + "=" * 60 + "🧠")
    print("  HASN AUTOMATED INTERNET TRAINING - SETUP")
    print("  Hierarchical Adaptive Spiking Network")
    print("🧠" + "=" * 60 + "🧠")
    print()

def check_python_version():
    """Check if Python version is compatible"""
    if sys.version_info < (3, 8):
        print("❌ Error: Python 3.8 or higher is required")
        print(f"   Current version: {sys.version}")
        sys.exit(1)
    else:
        print(f"✅ Python version: {sys.version.split()[0]}")

def install_dependencies():
    """Install required dependencies"""
    print("\n📦 Installing dependencies...")
    
    try:
        # Install requirements
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "-r", "requirements.txt"
        ])
        print("✅ Dependencies installed successfully")
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Error installing dependencies: {e}")
        print("   Try running: pip install -r requirements.txt")
        return False
    
    return True

def create_directories():
    """Create necessary directories"""
    print("\n📁 Creating directories...")
    
    directories = [
        "output",
        "logs",
        "data/cache"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"✅ Created: {directory}/")

def create_environment_file():
    """Create environment configuration file"""
    print("\n⚙️  Creating environment configuration...")
    
    env_content = """# HASN Automated Training Environment Configuration
# Customize these settings for your environment

# Training Settings
HASN_TRAINING_PROFILE=development
HASN_OUTPUT_DIR=output
HASN_LOG_LEVEL=INFO

# Data Collection Settings
HASN_REQUEST_DELAY=2.0
HASN_MAX_CONCURRENT=3
HASN_QUALITY_THRESHOLD=0.5

# Monitoring Settings
HASN_MONITOR_REFRESH=30
HASN_SAVE_INTERVAL=600

# Optional: Custom data sources (comma-separated URLs)
# HASN_CUSTOM_SOURCES=https://example.com/rss,https://example2.com/feed

# Optional: Content filters (comma-separated terms to avoid)
# HASN_CONTENT_FILTERS=advertisement,spam,clickbait
"""
    
    env_file = ".env.automated_training"
    
    if not os.path.exists(env_file):
        with open(env_file, 'w') as f:
            f.write(env_content)
        print(f"✅ Created: {env_file}")
    else:
        print(f"⚠️  Environment file already exists: {env_file}")

def test_core_imports():
    """Test that core modules can be imported"""
    print("\n🧪 Testing core imports...")
    
    tests = [
        ("numpy", "NumPy for numerical computing"),
        ("matplotlib", "Matplotlib for visualization"),
        ("networkx", "NetworkX for graph analysis"),
        ("aiohttp", "aiohttp for async web requests"),
        ("json", "JSON for data serialization")
    ]
    
    failed_imports = []
    
    for module, description in tests:
        try:
            __import__(module)
            print(f"✅ {module}: {description}")
        except ImportError:
            print(f"❌ {module}: {description} - FAILED")
            failed_imports.append(module)
    
    if failed_imports:
        print(f"\n⚠️  Some imports failed: {', '.join(failed_imports)}")
        print("   Try: pip install -r requirements.txt")
        return False
    
    return True

def test_brain_modules():
    """Test that brain modules can be imported"""
    print("\n🧠 Testing brain network modules...")
    
    # Add src to path
    sys.path.insert(0, 'src')
    
    brain_tests = [
        ("core.simplified_brain_network", "Core simplified brain architecture"),
        ("training.interactive_brain_trainer", "Interactive training system")
    ]
    
    failed_brain_imports = []
    
    for module, description in brain_tests:
        try:
            __import__(module)
            print(f"✅ {module}: {description}")
        except ImportError as e:
            print(f"❌ {module}: {description} - FAILED ({e})")
            failed_brain_imports.append(module)
    
    if failed_brain_imports:
        print(f"\n⚠️  Some brain modules failed: {', '.join(failed_brain_imports)}")
        print("   Check that you're running from the project root directory")
        return False
    
    return True

def create_quick_test():
    """Create a quick test script"""
    print("\n📝 Creating quick test script...")
    
    test_script = """#!/usr/bin/env python3
'''
Quick test for HASN Automated Training System
Run this to verify everything is working
'''

import sys
import os
import asyncio

# Add src to path
sys.path.insert(0, 'src')

async def test_training_system():
    print("🧠 Testing HASN Automated Training System...")
    
    try:
        # Test core imports
        from core.simplified_brain_network import SimpleBrainNetwork
        from training.automated_internet_trainer import AutomatedInternetTrainer, TrainingConfig
        
        print("✅ Core imports successful")
        
        # Test brain creation
        brain = SimpleBrainNetwork(num_neurons=50, connectivity_prob=0.1)
        print("✅ Brain network created")
        
        # Test configuration
        config = TrainingConfig(max_articles_per_session=5)
        trainer = AutomatedInternetTrainer(config)
        print("✅ Trainer configured")
        
        # Test pattern conversion
        pattern = trainer.converter.text_to_neural_pattern("test message", "test")
        print(f"✅ Neural pattern conversion: {len(pattern)} modules activated")
        
        print("\\n🎉 All tests passed! System is ready for automated training.")
        print("\\nNext steps:")
        print("   1. python src/training/train_cli.py start")
        print("   2. python src/training/train_cli.py monitor")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        print("\\nTroubleshooting:")
        print("   - Make sure you're in the project root directory")
        print("   - Check that requirements are installed: pip install -r requirements.txt")
        print("   - Verify Python version is 3.8+")

if __name__ == "__main__":
    asyncio.run(test_training_system())
"""
    
    test_file = "test_automated_training.py"
    with open(test_file, 'w') as f:
        f.write(test_script)
    
    # Make executable
    os.chmod(test_file, 0o755)
    print(f"✅ Created: {test_file}")

def print_quick_start():
    """Print quick start instructions"""
    print("\n🚀 SETUP COMPLETE!")
    print("=" * 50)
    print()
    print("Quick Start Commands:")
    print("=" * 20)
    print()
    print("1. Test the system:")
    print("   python test_automated_training.py")
    print()
    print("2. Start training (development mode):")
    print("   python src/training/train_cli.py start")
    print()
    print("3. Monitor progress:")
    print("   python src/training/train_cli.py monitor")
    print()
    print("4. Check status:")
    print("   python src/training/train_cli.py status")
    print()
    print("5. Generate report:")
    print("   python src/training/train_cli.py report")
    print()
    print("Advanced Usage:")
    print("=" * 15)
    print()
    print("• Production training:")
    print("  python src/training/train_cli.py start --profile production --continuous")
    print()
    print("• Custom configuration:")
    print("  Edit src/training/training_config.json")
    print()
    print("• Environment settings:")
    print("  Edit .env.automated_training")
    print()
    print("Documentation:")
    print("=" * 14)
    print("• Full guide: src/training/AUTOMATED_TRAINING_README.md")
    print("• Configuration: src/training/training_config.json")
    print("• Main project: docs/README.md")
    print()
    print("🧠 Ready to unleash continuous AI learning!")

def main():
    """Main setup function"""
    print_banner()
    
    # Check Python version
    check_python_version()
    
    # Install dependencies
    if not install_dependencies():
        sys.exit(1)
    
    # Create directories
    create_directories()
    
    # Create environment file
    create_environment_file()
    
    # Test imports
    if not test_core_imports():
        print("\n⚠️  Some dependencies are missing. Please install them before continuing.")
        sys.exit(1)
    
    # Test brain modules
    if not test_brain_modules():
        print("\n⚠️  Brain modules are not accessible. Check your setup.")
        sys.exit(1)
    
    # Create test script
    create_quick_test()
    
    # Print instructions
    print_quick_start()

if __name__ == "__main__":
    main()