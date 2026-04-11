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
        self.current_task = "easy_fraud_detection"
        self._rng = random.Random(42) # Default seed
        self._expected = "Approve"

    def reset(
        self, 
        task_id: Optional[str] = None,
        seed: Optional[int] = None, # Added required RL parameter
        **kwargs
    ) -> FraudTriageObservation:
        self._state = State(episode_id=str(uuid4()), step_count=0)
        
        if task_id:
            self.current_task = task_id
            
        # Re-initialize RNG for reproducible, dynamic generation
        if seed is not None:
            self._rng = random.Random(seed)

        return self._generate_dynamic_case(reward=0.01, done=False, feedback="Environment reset.")

    def _generate_dynamic_case(self, reward: float = 0.01, done: bool = False, feedback: str = "") -> FraudTriageObservation:
        # PROCEDURAL GENERATION: Infinite variations, deterministic grading.
        txn_id = f"TXN_{self._rng.randint(1000, 9999)}"
        
        if self.current_task == "easy_fraud_detection":
            # Easy: Obviously safe
            self._expected = "Approve"
            amount = round(self._rng.uniform(10.0, 150.0), 2)
            credit = self._rng.randint(700, 850)
            category = self._rng.choice(["Groceries", "Coffee", "Subscriptions"])
            chargebacks = False
            
        elif self.current_task == "medium_fraud_detection":
            # Medium: Borderline, requires flag
            self._expected = "Flag"
            amount = round(self._rng.uniform(300.0, 800.0), 2)
            credit = self._rng.randint(550, 680)
            category = self._rng.choice(["Electronics", "Travel", "Jewelry"])
            chargebacks = self._rng.choice([True, False])
            
        else: 
            # Hard: Obvious Fraud
            self._expected = "Reject"
            amount = round(self._rng.uniform(5000.0, 15000.0), 2)
            credit = self._rng.randint(300, 500)
            category = "Crypto"
            chargebacks = True

        return FraudTriageObservation(
            transaction_id=txn_id,
            amount=amount,
            merchant_category=category,
            credit_score=credit,
            has_chargebacks=chargebacks,
            done=done,
            reward=reward,
            metadata={"step": self._state.step_count, "feedback": feedback}
        )

    def step(self, action: FraudTriageAction) -> FraudTriageObservation:  # type: ignore[override]
        if self._state.step_count >= 1:
            return self._generate_dynamic_case(reward=0.01, done=True, feedback="Episode finished.")

        self._state.step_count += 1
        done = True 
        decision = getattr(action, "decision", "Invalid")

        # DETERMINISTIC GRADING (Rubric never changes)
        if decision == self._expected:
            reward = 0.99
            feedback = f"Perfect! Expected {self._expected}."
        elif decision == "Flag" and self._expected != "Flag":
            reward = 0.50
            feedback = "Partial credit. Flagged safely but a definitive answer was expected."
        else:
            reward = 0.01
            feedback = f"Incorrect. Expected {self._expected}."
            
        obs = self._generate_dynamic_case(reward=reward, done=done, feedback=feedback)
        obs.metadata["agent_reasoning"] = getattr(action, "reasoning", "None")
        obs.metadata["score"] = reward
        
        return obs

    @property
    def state(self) -> State:
        return self._state