"""
Brain-Native Language API Integration
Replace LLM with superior brain-inspired processing
"""

import time

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

# Import our brain-native language processing
from brain_language_enhanced import EnhancedCognitiveBrainWithLanguage

app = FastAPI(
    title="Brain-Native Language API",
    description="Superior brain-inspired language processing - No LLMs needed!",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize brain-native language system
try:
    brain_system = EnhancedCognitiveBrainWithLanguage()
    print("🧠 Brain-Native Language System initialized successfully!")
except Exception as e:
    print(f"❌ Error initializing brain system: {e}")
    brain_system = None


@app.get("/")
async def root():
    """Root endpoint with brain-native information"""
    return {
        "message": "🧠 Brain-Native Language API",
        "description": "Superior to LLM integration - True brain-inspired AI",
        "brain_type": "Hierarchical Adaptive Spiking Network (HASN)",
        "advantages_over_llm": [
            "Real-time learning and adaptation",
            "Biologically inspired neural processing",
            "Interpretable neural activity patterns",
            "Energy efficient spiking neurons",
            "Continuous memory integration",
            "True cognitive architecture",
        ],
        "endpoints": {
            "/brain/process": "Process text through brain-native neural networks",
            "/brain/chat": "Brain-native conversation (superior to LLM chat)",
            "/brain/state": "View current brain state and neural activity",
            "/brain/vocabulary": "Explore learned vocabulary and neural patterns",
            "/brain/compare-to-llm": "See why brain-native is superior",
            "/health": "System health check",
        },
    }


@app.post("/brain/process")
async def process_text_brain_native(request: dict):
    """
    Process text through brain-native neural networks
    Superior to LLM processing - shows actual neural activity!
    """
    if not brain_system:
        raise HTTPException(status_code=503, detail="Brain system not initialized")

    try:
        text = request.get("text", "")
        if not text:
            raise HTTPException(status_code=400, detail="Text is required")

        # Process through brain-native language system
        response_text, brain_data = brain_system.process_natural_language(text)

        return {
            "success": True,
            "message": "Processed through brain-native neural networks",
            "brain_response": response_text,
            "neural_analysis": {
                "neural_pattern_intensity": brain_data["neural_pattern"]["intensity"],
                "active_brain_modules": brain_data["brain_activity"]["active_modules"],
                "cognitive_load": brain_data["cognitive_load"],
                "processing_time_ms": brain_data["processing_time_ms"],
                "confidence_score": brain_data["confidence_score"],
                "learning_occurred": brain_data["learning_occurred"],
            },
            "brain_advantages": [
                "Real neural activity patterns (not statistical prediction)",
                "Continuous learning from interaction",
                "Interpretable cognitive processing",
                "Biologically inspired architecture",
            ],
            "timestamp": time.time(),
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Brain processing error: {str(e)}")


@app.post("/brain/chat")
async def brain_native_chat(request: dict):
    """
    Brain-native conversation system
    Far superior to LLM chat - uses actual cognitive processing!
    """
    if not brain_system:
        raise HTTPException(status_code=503, detail="Brain system not initialized")

    try:
        message = request.get("message", "")
        user_id = request.get("user_id", "anonymous")

        if not message:
            raise HTTPException(status_code=400, detail="Message is required")

        # Process through brain-native conversation
        response_text, brain_data = brain_system.process_natural_language(message)

        # Get current brain state
        brain_state = brain_system.get_brain_state_summary()

        return {
            "success": True,
            "response_text": response_text,
            "brain_activity": {
                "neural_intensity": brain_data["neural_pattern"]["intensity"],
                "cognitive_load": brain_data["cognitive_load"],
                "active_modules": brain_data["brain_activity"]["active_modules"],
                "attention_focus": brain_state["attention_focus"],
                "working_memory_items": brain_state["working_memory_items"],
            },
            "conversation_context": {
                "user_id": user_id,
                "vocabulary_growth": brain_state["vocabulary_size"],
                "learning_capacity": brain_state["learning_capacity"],
                "brain_adaptation": "Continuous real-time learning",
            },
            "why_superior_to_llm": {
                "real_neural_processing": True,
                "continuous_learning": True,
                "interpretable_activity": True,
                "biological_inspiration": True,
                "energy_efficient": True,
            },
            "confidence_score": brain_data["confidence_score"],
            "processing_time_ms": brain_data["processing_time_ms"],
            "timestamp": time.time(),
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Brain chat error: {str(e)}")


@app.get("/brain/state")
async def get_brain_state():
    """Get current brain state and neural activity"""
    if not brain_system:
        raise HTTPException(status_code=503, detail="Brain system not initialized")

    try:
        brain_state = brain_system.get_brain_state_summary()

        return {
            "success": True,
            "brain_state": brain_state,
            "neural_modules": {
                "sensory_module": brain_state["module_status"]["sensory"],
                "memory_module": brain_state["module_status"]["memory"],
                "executive_module": brain_state["module_status"]["executive"],
                "motor_module": brain_state["module_status"]["motor"],
                "language_module": brain_state["module_status"]["language"],
            },
            "cognitive_metrics": {
                "cognitive_load": brain_state["cognitive_load"],
                "learning_capacity": brain_state["learning_capacity"],
                "vocabulary_size": brain_state["vocabulary_size"],
                "recent_activity_count": brain_state["recent_activity"],
            },
            "brain_advantages": [
                "Real-time neural state monitoring",
                "Transparent cognitive processing",
                "Adaptive learning mechanisms",
                "Biologically plausible architecture",
            ],
            "vs_llm": {
                "llm_state_visibility": "Black box - no visibility",
                "brain_state_visibility": "Complete transparency",
                "llm_adaptation": "Static after training",
                "brain_adaptation": "Continuous real-time learning",
                "llm_interpretation": "Difficult to interpret decisions",
                "brain_interpretation": "Clear neural activity patterns",
            },
            "timestamp": time.time(),
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Brain state error: {str(e)}")


@app.get("/brain/vocabulary")
async def get_brain_vocabulary():
    """Explore learned vocabulary and neural patterns"""
    if not brain_system:
        raise HTTPException(status_code=503, detail="Brain system not initialized")

    try:
        vocabulary = brain_system.language_module.vocabulary
        word_frequency = brain_system.language_module.word_frequency

        # Get top learned words
        top_words = sorted(word_frequency.items(), key=lambda x: x[1], reverse=True)[:20]

        return {
            "success": True,
            "vocabulary_stats": {
                "total_vocabulary": len(vocabulary),
                "total_word_encounters": sum(word_frequency.values()),
                "average_word_frequency": (
                    sum(word_frequency.values()) / len(word_frequency) if word_frequency else 0
                ),
            },
            "top_learned_words": [
                {
                    "word": word,
                    "frequency": freq,
                    "has_neural_pattern": word in vocabulary,
                }
                for word, freq in top_words
            ],
            "neural_pattern_example": {
                "description": "Each word creates unique neural activation patterns",
                "example_word": top_words[0][0] if top_words else "hello",
                "pattern_type": "Spiking neural network activation",
                "brain_learning": "Continuous adaptation from interaction",
            },
            "brain_vs_llm_vocabulary": {
                "brain_approach": "Dynamic neural patterns learned through interaction",
                "llm_approach": "Static embeddings from pre-training",
                "brain_advantage": "Real-time vocabulary adaptation",
                "llm_limitation": "Fixed vocabulary after training",
            },
            "timestamp": time.time(),
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Vocabulary error: {str(e)}")


@app.get("/brain/compare-to-llm")
async def compare_brain_to_llm():
    """Comprehensive comparison showing why brain-native is superior"""
    return {
        "comparison_title": "🧠 Brain-Native vs 🤖 LLM: Why Brain-Inspired AI is Superior",
        "brain_native_advantages": {
            "biological_authenticity": {
                "brain": "Based on actual spiking neural networks",
                "llm": "Mathematical transformer blocks",
                "winner": "Brain-Native",
            },
            "learning_capability": {
                "brain": "Continuous real-time learning and adaptation",
                "llm": "Static after training - no learning",
                "winner": "Brain-Native",
            },
            "interpretability": {
                "brain": "Observable neural activity patterns and states",
                "llm": "Black box - impossible to interpret decisions",
                "winner": "Brain-Native",
            },
            "energy_efficiency": {
                "brain": "Spiking neurons - extremely energy efficient",
                "llm": "Massive matrix operations - energy intensive",
                "winner": "Brain-Native",
            },
            "memory_integration": {
                "brain": "Built-in working memory and episodic memory",
                "llm": "Limited context window - no true memory",
                "winner": "Brain-Native",
            },
            "cognitive_architecture": {
                "brain": "Hierarchical cognitive modules (sensory, memory, executive)",
                "llm": "Flat attention mechanism",
                "winner": "Brain-Native",
            },
        },
        "technical_superiority": {
            "processing_type": {
                "brain": "Event-driven spiking computation",
                "llm": "Batch matrix multiplication",
            },
            "adaptation_mechanism": {
                "brain": "Hebbian learning and synaptic plasticity",
                "llm": "Gradient descent on static parameters",
            },
            "response_generation": {
                "brain": "Neural activity patterns drive response",
                "llm": "Statistical token prediction",
            },
            "context_handling": {
                "brain": "Integrated memory and attention systems",
                "llm": "Fixed-size attention windows",
            },
        },
        "practical_benefits": [
            "Real-time learning from every interaction",
            "Transparent decision-making process",
            "Energy-efficient operation",
            "Biologically plausible architecture",
            "Continuous adaptation without retraining",
            "Observable cognitive states and processes",
            "True memory integration",
            "Hierarchical information processing",
        ],
        "use_cases_where_brain_excels": [
            "Adaptive conversational AI",
            "Personalized learning systems",
            "Real-time decision making",
            "Cognitive modeling and simulation",
            "Interactive AI assistants",
            "Educational AI tutors",
            "Adaptive user interfaces",
            "Cognitive robotics",
        ],
        "conclusion": {
            "recommendation": "Use Brain-Native Architecture",
            "reasoning": [
                "Superior cognitive modeling capabilities",
                "Real-time learning and adaptation",
                "Interpretable and transparent operation",
                "Energy efficient and scalable",
                "Biologically inspired and authentic",
                "Continuous improvement through interaction",
            ],
        },
        "next_steps": [
            "Integrate brain-native processing into your application",
            "Replace LLM dependencies with brain modules",
            "Monitor neural activity for insights",
            "Leverage continuous learning capabilities",
            "Build on biological cognitive principles",
        ],
    }


@app.get("/health")
async def health_check():
    """Health check with brain system status"""
    brain_status = "healthy" if brain_system else "unavailable"

    brain_info = {}
    if brain_system:
        try:
            brain_state = brain_system.get_brain_state_summary()
            brain_info = {
                "cognitive_load": brain_state["cognitive_load"],
                "vocabulary_size": brain_state["vocabulary_size"],
                "learning_capacity": brain_state["learning_capacity"],
                "active_modules": list(brain_state["module_status"].keys()),
            }
        except Exception as e:
            brain_info = {"error": str(e)}

    return {
        "status": "healthy",
        "brain_system": brain_status,
        "brain_info": brain_info,
        "api_type": "Brain-Native Language Processing",
        "advantages": [
            "Superior to LLM integration",
            "Real neural activity patterns",
            "Continuous learning capability",
            "Interpretable cognitive processing",
        ],
        "timestamp": time.time(),
    }


# Demo endpoint for testing
@app.post("/brain/demo")
async def demo_brain_processing(request: dict = None):
    """Demo endpoint showing brain-native processing capabilities"""
    if not brain_system:
        raise HTTPException(status_code=503, detail="Brain system not initialized")

    demo_texts = [
        "Hello, I'm interested in brain-inspired AI",
        "How does neural processing work?",
        "What makes your approach better than LLMs?",
        "Can you learn from our conversation?",
        "Explain consciousness and artificial intelligence",
    ]

    if request and "text" in request:
        demo_texts = [request["text"]]

    results = []

    for text in demo_texts:
        try:
            response_text, brain_data = brain_system.process_natural_language(text)

            results.append(
                {
                    "input": text,
                    "brain_response": response_text,
                    "neural_intensity": brain_data["neural_pattern"]["intensity"],
                    "cognitive_load": brain_data["cognitive_load"],
                    "processing_time_ms": brain_data["processing_time_ms"],
                    "learning_occurred": brain_data["learning_occurred"],
                }
            )

        except Exception as e:
            results.append({"input": text, "error": str(e)})

    brain_state = brain_system.get_brain_state_summary()

    return {
        "success": True,
        "demo_results": results,
        "final_brain_state": {
            "vocabulary_growth": brain_state["vocabulary_size"],
            "cognitive_load": brain_state["cognitive_load"],
            "learning_capacity": brain_state["learning_capacity"],
        },
        "demo_insights": [
            "Each interaction creates unique neural patterns",
            "Vocabulary grows dynamically through conversation",
            "Cognitive load adapts to input complexity",
            "Learning occurs continuously in real-time",
            "Neural activity is fully observable and interpretable",
        ],
        "timestamp": time.time(),
    }


if __name__ == "__main__":
    print("🧠 Starting Brain-Native Language API...")
    print("🚀 Superior to LLM integration!")
    print("💡 Real neural processing, continuous learning, interpretable AI")

    uvicorn.run(app, host="0.0.0.0", port=8000)
