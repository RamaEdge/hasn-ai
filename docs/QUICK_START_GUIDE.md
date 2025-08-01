# 🧠 Brain-Inspired Neural Network - Quick Start Guide

## 🚀 Getting Started

### Prerequisites
- Python 3.8+ 
- Virtual environment (recommended)

### Installation

1. **Clone the repository:**
```bash
git clone <repository-url>
cd cde-hack-session
```

2. **Set up virtual environment:**
```bash
# Create virtual environment (if not exists)
python3 -m venv .venv

# Activate virtual environment
source .venv/bin/activate  # On macOS/Linux
# or
.venv\Scripts\activate     # On Windows
```

3. **Install dependencies:**
```bash
pip install numpy networkx
# Optional: pip install matplotlib (for visualizations)
```

## 🎯 Running the Demonstrations

### 1. Simple Brain Demo (Recommended Start)
```bash
python src/simple_brain_demo.py
```
**What it shows:**
- ⚡ Spiking neural dynamics
- 🧠 Learning and adaptation
- 🎯 Attention mechanisms
- 💾 Working memory utilization

### 2. Advanced Brain Network
```bash
python src/advanced_brain_network.py
```
**What it shows:**
- 🔬 Cognitive capabilities
- 🌱 Structural plasticity
- 🔄 Memory consolidation
- 📊 Multi-scale temporal processing

### 3. Comprehensive Analysis
```bash
python src/demo_and_analysis.py
```
**What it shows:**
- 📈 Performance comparisons
- 📊 Detailed analysis reports
- 🎯 Brain vs. traditional AI comparison

## 📊 Understanding the Results

### Output Files
After running demos, check the `output/` directory:

- `simple_brain_demo_results.json` - Quantitative results
- `brain_network_analysis_report.md` - Technical analysis
- `analysis_results.json` - Raw experimental data

### Key Metrics to Watch

**🧠 Plasticity Score (0-1):**
- 0.7+ = Excellent synaptic adaptation
- 0.5-0.7 = Good learning capability
- <0.5 = Limited plasticity

**🎯 Attention Stability (0-1):**
- 0.9+ = Highly stable selective processing
- 0.7-0.9 = Good attention control
- <0.7 = Dynamic/unstable attention

**💾 Memory Usage:**
- 7/7 = Full working memory utilization
- 5-6 = Good memory engagement
- <5 = Underutilized memory capacity

## 🔧 Customization

### Modify Network Architecture
```python
# In simple_brain_demo.py
network = SimpleBrainNetwork([30, 25, 20, 15])  # Module sizes
#                            ↑   ↑   ↑   ↑
#                        sensory memory exec motor
```

### Adjust Learning Parameters
```python
# In brain neurons
learning_rate = 0.01      # STDP strength
noise_level = 0.1         # Neural noise
connectivity_prob = 0.1   # Connection probability
```

### Create Custom Input Patterns
```python
# Example: Rhythmic input pattern
def custom_input_pattern(step):
    if step % 50 < 10:  # Active every 50 steps for 10 steps
        return {0: {i: True for i in range(5)}}  # Activate first 5 neurons in module 0
    return {}
```

## 🧪 Experimental Modifications

### 1. Test Different Timescales
```python
# Modify neuron parameters
params.tau_m = 20.0        # Membrane time constant (ms)
params.tau_adapt = 100.0   # Adaptation time constant (ms)
```

### 2. Experiment with Connectivity
```python
# Change connection probabilities
internal_connectivity = 0.15    # Within-module connections
inter_module_connectivity = 0.05 # Between-module connections
```

### 3. Adjust Memory Capacity
```python
# Modify working memory
memory_capacity = 10       # Increase from default 7
decay_rate = 0.005        # Slower forgetting
```

## 🐛 Troubleshooting

### Common Issues

**ImportError: No module named 'numpy'**
```bash
# Solution: Install dependencies
pip install numpy networkx
```

**No output files generated**
```bash
# Solution: Check output directory exists
mkdir -p output
```

**Low plasticity scores**
```bash
# Solution: Increase simulation time or learning rate
learning_rate = 0.02  # Increase from 0.01
num_steps = 2000     # Increase from 1000
```

**Memory not being used**
```bash
# Solution: Use more significant input patterns
# Increase input strength or pattern complexity
```

## 📚 Learn More

### Key Files to Explore
1. **`src/simple_brain_demo.py`** - Start here for basic concepts
2. **`src/brain_inspired_network.py`** - Core architecture
3. **`src/advanced_brain_network.py`** - Advanced features
4. **`BRAIN_INSPIRED_RESEARCH.md`** - Scientific background

### Understanding the Science
- **STDP**: Spike-timing dependent plasticity
- **LIF**: Leaky integrate-and-fire neurons  
- **Homeostasis**: Activity regulation mechanisms
- **Working Memory**: Capacity-limited storage system

## 🎉 Success Indicators

You'll know it's working when you see:
- ✅ Neurons spiking in response to inputs
- ✅ Synaptic weights changing over time
- ✅ Attention focusing on active modules
- ✅ Working memory filling up during activity
- ✅ Network adapting to input patterns

## 🚀 Next Steps

1. **Understand the basics** with simple demo
2. **Explore parameters** by modifying values
3. **Create custom patterns** for specific tasks
4. **Analyze results** using generated reports
5. **Extend architecture** with new capabilities

---

*Happy experimenting with brain-inspired AI! 🧠✨*
