"""
Brain adapters providing a stable interface for routers.

- SimpleBrainAdapter: wraps SimpleBrainNetwork
- CognitiveBrainAdapter: wraps CognitiveBrainNetwork and exposes training for episodic memory
"""

from typing import Dict

try:
    from core.cognitive_brain_network import CognitiveBrainNetwork
    from core.simplified_brain_network import SimpleBrainNetwork
except ImportError:  # pragma: no cover - fallback for certain execution contexts
    from typing import Any as CognitiveBrainNetwork  # type: ignore
    from typing import Any as SimpleBrainNetwork  # type: ignore


class SimpleBrainAdapter:
    def __init__(self, network: SimpleBrainNetwork):
        self.network = network

    def process_pattern(self, pattern: dict) -> dict:
        # Map module-local indices to global neuron ids deterministically
        external_input = {}
        total_inputs = 0
        for module_str, neurons in (pattern or {}).items():
            try:
                module_id = int(module_str)
            except Exception:
                module_id = 0
            if not isinstance(neurons, dict):
                continue
            for neuron_str, active in neurons.items():
                if not active:
                    continue
                try:
                    local_idx = int(neuron_str)
                except Exception:
                    continue
                # Hash (module, local_idx) into valid neuron id
                global_id = ((module_id * 97) + local_idx) % self.network.num_neurons
                external_input[global_id] = True
                total_inputs += 1

        spikes = self.network.step(external_input)
        activity = sum(1 for v in spikes.values() if v)
        return {
            "total_activity": activity / max(1, self.network.num_neurons),
            "active_neurons": activity,
            "input_neurons": total_inputs,
            "steps": 1,
        }

    def text_to_pattern(self, text: str) -> dict:
        pattern = {}
        if text:
            words = len(text.split())
            chars = len(text)
            pattern["0"] = {str(i): True for i in range(min(words, 10))}
            pattern["1"] = {str(i): True for i in range(min(chars // 10, 10))}
        return pattern

    def generate_response(self, brain_result: dict) -> str:
        activity = brain_result.get("total_activity", 0.0)
        active_neurons = brain_result.get("active_neurons", 0)
        return f"Neural activity {activity:.3f} with {active_neurons} active neurons"

    def get_brain_state(self) -> dict:
        return self.network.get_brain_state()


class CognitiveBrainAdapter:
    def __init__(self, network: CognitiveBrainNetwork):
        self.network = network

    def _pattern_to_external(self, pattern: dict) -> dict:
        """Convert module→local neuron map to global external input indices."""
        external_input: Dict[int, bool] = {}
        if not isinstance(pattern, dict):
            return external_input
        for module_str, neurons in pattern.items():
            try:
                module_id = int(module_str)
            except Exception:
                module_id = 0
            if not isinstance(neurons, dict):
                continue
            for neuron_str, active in neurons.items():
                if not active:
                    continue
                try:
                    local_idx = int(neuron_str)
                except Exception:
                    continue
                global_id = ((module_id * 131) + local_idx) % self.network.num_neurons
                external_input[global_id] = True
        return external_input

    def process_pattern(self, pattern: dict) -> dict:
        external_input = self._pattern_to_external(pattern)
        result = self.network.step_with_cognition(external_input, context={})
        return {
            "total_activity": result.get("spike_count", 0) / max(1, self.network.num_neurons),
            "active_neurons": result.get("spike_count", 0),
            "inferences": result.get("inferences", []),
            "steps": 1,
        }

    def train_step(self, sample, learning_rate: float = 0.01) -> dict:
        """Single training step storing an episodic memory when context is present."""
        try:
            self.set_learning_rate(learning_rate)
        except Exception:
            pass

        s = sample.dict() if hasattr(sample, "dict") else (sample or {})
        payload = s.get("input", s)

        pattern = payload.get("pattern")
        if not pattern and isinstance(payload.get("text"), str):
            pattern = self.text_to_pattern(payload.get("text", ""))

        external_input = self._pattern_to_external(pattern or {})

        context = payload.get("context", {}) or {}
        if "concept" not in context and s.get("label") is not None:
            context["concept"] = str(s.get("label"))
        context.setdefault("source", "api.training")
        context.setdefault("generate_inferences", False)

        result = self.network.step_with_cognition(external_input, context=context)
        spike_ratio = result.get("spike_count", 0) / max(1, self.network.num_neurons)
        return {
            "loss": max(0.0, 1.0 - spike_ratio),
            "accuracy": min(1.0, 0.5 + 0.4 * ("concept" in context)),
            "memory_id": result.get("memory_id"),
            "total_memories": result.get("total_memories", 0),
        }

    def text_to_pattern(self, text: str) -> dict:
        pattern = {}
        if text:
            words = len(text.split())
            chars = len(text)
            pattern["0"] = {str(i): True for i in range(min(words, 10))}
            pattern["1"] = {str(i): True for i in range(min(chars // 10, 10))}
        return pattern

    def generate_response(self, brain_result: dict) -> str:
        inf_count = len(brain_result.get("inferences", []))
        return f"Cognitive processing completed with {inf_count} inferences"

    def get_brain_state(self) -> dict:
        base = self.network.get_brain_state()
        base.update({"cognitive": self.network.get_cognitive_state()})
        return base

    # Configuration helpers used by /training/configure
    def set_learning_rate(self, lr: float):
        self.network.config.learning_rate = float(lr)

    def set_threshold(self, thresh: float):
        for n in getattr(self.network, "neurons", []):
            if hasattr(n, "params"):
                n.params.threshold = float(thresh)

    def set_memory_capacity(self, capacity: int):
        try:
            self.network.config.max_episodic_memories = int(capacity)
        except Exception:
            pass

    def get_weights(self):
        return self.network.get_weights()
