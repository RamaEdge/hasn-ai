# 🧠 HASN-AI: Automated Internet Training System

**Revolutionary brain-inspired AI that learns continuously from the internet**

This project implements a Hierarchical Adaptive Spiking Network (HASN) with **automated internet training capabilities**. The system can autonomously collect information from web sources and train itself in real-time, mimicking how biological brains learn from their environment.

## 🚀 Quick Start - Automated Training

### 1. Setup
```bash
python setup_automated_training.py
```

### 2. Start Training
```bash
# Basic training
python src/training/train_cli.py start

# Continuous training (runs 24/7)
python src/training/train_cli.py start --profile production --continuous
```

### 3. Monitor Progress
```bash
# Real-time monitoring
python src/training/train_cli.py monitor

# Generate reports
python src/training/train_cli.py report
```

### 4. API Access
```bash
# Start API server
python src/api/main.py

# Visit http://localhost:8000/docs for full API
```

## ✨ What Makes This Revolutionary

- **🌐 Self-Learning**: Automatically learns from internet content 24/7
- **🧠 Brain-Inspired**: Uses actual spiking neural networks, not transformers
- **⚡ Real-Time Adaptation**: Continuously updates knowledge as it learns
- **📊 Full Observability**: Watch exactly what the brain is learning
- **🎯 Quality-Driven**: Intelligently filters and prioritizes content
- **🔄 Memory Consolidation**: Simulates sleep cycles for memory strengthening

For detailed documentation, see [Automated Training Guide](src/training/AUTOMATED_TRAINING_README.md)
