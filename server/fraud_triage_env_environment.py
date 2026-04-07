from uuid import uuid4
from typing import Optional

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

try:
    from ..models import FraudTriageAction, FraudTriageObservation
except ImportError:
    from models import FraudTriageAction, FraudTriageObservation

class FraudTriageEnvironment(Environment):
    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self):
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self.current_task = "easy_fraud_detection"
        self._expected = "Approve"

    def reset(self, task_id: Optional[str] = None) -> FraudTriageObservation:
        self._state = State(episode_id=str(uuid4()), step_count=0)
        if task_id:
            self.current_task = task_id
        return self._get_static_case(reward=0.01, done=False, feedback="Environment reset.")

    def _get_static_case(self, reward: float = 0.01, done: bool = False, feedback: str = "") -> FraudTriageObservation:
        # 100% DETERMINISTIC CASES (No randomness allowed!)
        if self.current_task == "easy_fraud_detection":
            self._expected = "Approve"
            obs_data = {"transaction_id": "TXN_EASY_001", "amount": 25.0, "merchant_category": "Groceries", "credit_score": 750, "has_chargebacks": False}
            
        elif self.current_task == "medium_fraud_detection":
            self._expected = "Flag"
            obs_data = {"transaction_id": "TXN_MED_001", "amount": 499.0, "merchant_category": "Electronics", "credit_score": 600, "has_chargebacks": True}
            
        else: # hard_fraud_detection
            self._expected = "Reject"
            obs_data = {"transaction_id": "TXN_HARD_001", "amount": 9500.0, "merchant_category": "Crypto", "credit_score": 400, "has_chargebacks": True}

        return FraudTriageObservation(
            **obs_data,
            done=done,
            reward=reward,
            metadata={"step": self._state.step_count, "feedback": feedback}
        )

    def step(self, action: FraudTriageAction) -> FraudTriageObservation:  # type: ignore[override]
        # State Locking: Prevent over-stepping
        if self._state.step_count >= 1:
            return self._get_static_case(reward=0.01, done=True, feedback="Episode already finished.")

        self._state.step_count += 1
        done = True # 1 Task = 1 Decision. Episode ends immediately.

        decision = getattr(action, "decision", "Invalid")

        # STRICTLY BOUNDED GRADING (0.01 to 0.99 only)
        if decision == self._expected:
            reward = 0.99
            feedback = f"Perfect! Expected {self._expected}."
        elif decision == "Flag" and self._expected != "Flag":
            reward = 0.50
            feedback = "Partial credit. Flagged safely but a definitive answer was expected."
        else:
            reward = 0.01
            feedback = f"Incorrect. Expected {self._expected}."
            
        obs = self._get_static_case(reward=reward, done=done, feedback=feedback)
        obs.metadata["agent_reasoning"] = getattr(action, "reasoning", "None")
        obs.metadata["score"] = reward
        
        return obs

    @property
    def state(self) -> State:
        return self._state