from typing import Dict
from openenv.core import EnvClient
from openenv.core.client_types import StepResult
from openenv.core.env_server.types import State
try:
    from .models import FraudTriageAction, FraudTriageObservation
except ImportError:
    from models import FraudTriageAction, FraudTriageObservation

class FraudTriageEnv(EnvClient[FraudTriageAction, FraudTriageObservation, State]):
    
    def _step_payload(self, action: FraudTriageAction) -> Dict:
        return {
            "decision": action.decision,
            "reasoning": action.reasoning,
        }

    def _parse_result(self, payload: Dict) -> StepResult[FraudTriageObservation]:
        obs_data = payload.get("observation", {})
        
        # Dynamically unpack the observation data instead of hardcoding fields
        observation = FraudTriageObservation(**obs_data)

        return StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict) -> State:
        return State(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
        )