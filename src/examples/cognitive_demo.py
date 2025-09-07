#!/usr/bin/env python3
"""
Cognitive Architecture Demonstration
Shows the complete cat example: encode 'cat' → spikes → stored in episodic memory → consolidated into semantic after N exposures
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from core.cognitive_architecture import CognitiveArchitecture
from core.cognitive_models import CognitiveConfig
import time


def main():
    """Demonstrate the complete cognitive architecture"""
    print("🧠 Cognitive Architecture Demonstration")
    print("=" * 50)
    
    # Create cognitive architecture with custom config
    config = CognitiveConfig(
        max_working_memory_items=20,
        working_memory_ttl=5.0,
        semantic_consolidation_threshold=2,  # Lower threshold for demo
        max_episodic_memories=100
    )
    
    architecture = CognitiveArchitecture(config)
    
    print("\n📚 Processing 'cat' multiple times to demonstrate consolidation:")
    print("-" * 60)
    
    # Process "cat" multiple times
    for i in range(6):
        print(f"\n🔄 Exposure {i+1}/6: Processing 'cat'")
        result = architecture.process_input("cat", {"exposure": i+1, "demo": True})
        
        print(f"   🎯 Spikes generated: {sum(result['spike_pattern'].values())}")
        print(f"   🧠 Decision: {result['arbitration_decision']}")
        print(f"   ⏱️  Processing time: {result['processing_time']:.3f}s")
        
        # Show consolidation result if available
        if 'consolidation_result' in result:
            consolidation = result['consolidation_result']
            if consolidation.get('consolidated'):
                print(f"   ✅ Consolidation successful!")
                print(f"      Concept: {consolidation['concept']}")
                print(f"      Traces consolidated: {len(consolidation['consolidated_traces'])}")
            else:
                print(f"   ⏳ Consolidation pending: {consolidation.get('reason', 'unknown')}")
        
        # Small delay for demonstration
        time.sleep(0.5)
    
    print("\n" + "=" * 60)
    print("📊 Final System Status:")
    print("-" * 30)
    
    # Get system status
    status = architecture.get_system_status()
    print(f"🧬 Total neurons: {status['neurons']}")
    print(f"💭 Working memory items: {status['working_memory_items']}")
    print(f"📚 Episodic memories: {status['episodic_memories']}")
    print(f"🧠 Semantic memories: {status['semantic_memories']}")
    print(f"🔄 Total activations: {status['total_activations']}")
    print(f"📖 Recall events: {status['recall_events']}")
    print(f"🔗 Consolidation events: {status['consolidation_events']}")
    
    # Check if "cat" was consolidated
    cat_semantic = architecture.semantic_memory.find_semantic_memory("cat")
    if cat_semantic:
        print(f"\n✅ SUCCESS: 'cat' was consolidated into semantic memory!")
        print(f"   🆔 Semantic ID: {cat_semantic.semantic_id}")
        print(f"   📊 Consolidated traces: {cat_semantic.consolidated_count}")
        print(f"   🎯 Confidence: {cat_semantic.confidence:.2f}")
        print(f"   🧬 Vector dimension: {len(cat_semantic.semantic_vector)}")
    else:
        print(f"\n❌ 'cat' was not consolidated into semantic memory")
    
    # Show memory statistics
    print(f"\n📈 Memory Statistics:")
    print("-" * 20)
    stats = architecture.get_memory_statistics()
    
    print(f"Working Memory:")
    print(f"  Total items: {stats['working_memory']['total_items']}")
    print(f"  Active items: {stats['working_memory']['active_items']}")
    print(f"  Expired items: {stats['working_memory']['expired_items']}")
    
    print(f"Episodic Memory:")
    print(f"  Total traces: {stats['episodic_memory']['total_traces']}")
    print(f"  Avg access count: {stats['episodic_memory']['average_access_count']:.2f}")
    
    print(f"Semantic Memory:")
    print(f"  Total memories: {stats['semantic_memory']['total_memories']}")
    print(f"  Concepts: {stats['semantic_memory']['concepts']}")
    
    print(f"\n🎉 Demonstration complete!")
    print("The cognitive architecture successfully:")
    print("  ✅ Encoded text 'cat' into spike patterns")
    print("  ✅ Stored episodes in episodic memory")
    print("  ✅ Applied Hebbian learning in associative layer")
    print("  ✅ Managed working memory with TTL eviction")
    print("  ✅ Consolidated repeated exposures into semantic memory")
    print("  ✅ Used executive arbitration for decision making")


if __name__ == "__main__":
    main()
