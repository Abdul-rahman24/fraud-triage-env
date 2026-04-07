from uuid import uuid4
import random
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
        self.max_steps = 3
        self.current_task = "easy_fraud_detection"
        self._current_truth = "Approve"

    def reset(self, task_id: Optional[str] = None) -> FraudTriageObservation:
        self._state = State(episode_id=str(uuid4()), step_count=0)
        if task_id:
            self.current_task = task_id
        return self._generate_case()

    def _generate_case(self) -> FraudTriageObservation:
        is_fraud = random.choice([True, False])
        
        if self.current_task == "easy_fraud_detection":
            if is_fraud:
                self._current_truth = "Reject"
                obs_data = {"transaction_id": f"TXN_E_{self._state.step_count}", "amount": 9500.0, "merchant_category": "Crypto", "credit_score": 400, "has_chargebacks": True}
            else:
                self._current_truth = "Approve"
                obs_data = {"transaction_id": f"TXN_E_{self._state.step_count}", "amount": 25.0, "merchant_category": "Groceries", "credit_score": 750, "has_chargebacks": False}
                
        elif self.current_task == "medium_fraud_detection":
            if is_fraud:
                self._current_truth = "Reject"
                obs_data = {"transaction_id": f"TXN_M_{self._state.step_count}", "amount": 150.0, "merchant_category": "Electronics", "credit_score": 600, "has_chargebacks": False}
            else:
                self._current_truth = "Approve"
                obs_data = {"transaction_id": f"TXN_M_{self._state.step_count}", "amount": 45.0, "merchant_category": "Travel", "credit_score": 710, "has_chargebacks": False}
                
        else: # hard_fraud_detection
            is_flag = random.choice([True, False])
            if is_flag:
                self._current_truth = "Flag"
                obs_data = {"transaction_id": f"TXN_H_{self._state.step_count}", "amount": 499.0, "merchant_category": "Retail", "credit_score": 650, "has_chargebacks": True}
            else:
                self._current_truth = "Approve"
                obs_data = {"transaction_id": f"TXN_H_{self._state.step_count}", "amount": 89.0, "merchant_category": "Retail", "credit_score": 800, "has_chargebacks": False}

        return FraudTriageObservation(
            **obs_data,
            done=False,
            reward=0.0, 
            metadata={"step": self._state.step_count}
        )

    def step(self, action: FraudTriageAction) -> FraudTriageObservation:  # type: ignore[override]
        self._state.step_count += 1
        
        # PROPER REINFORCEMENT LEARNING SCALING
        # 3 steps * 0.33 = 0.99 Total Score
        if action.decision == self._current_truth:
            reward = 0.33  
            feedback = f"Correct! Expected {self._current_truth}."
        elif action.decision == "Flag":
            reward = 0.15
            feedback = f"Partial credit. Flagged safely, but {self._current_truth} was expected."
        else:
            reward = 0.01  
            feedback = f"Incorrect. Expected {self._current_truth}."

        done = self._state.step_count >= self.max_steps
        
        next_obs = self._generate_case()
        next_obs.done = done
        next_obs.reward = reward
        next_obs.metadata = {"feedback": feedback, "agent_reasoning": action.reasoning}
        
        return next_obs

    @property
    def state(self) -> State:
        return self._state