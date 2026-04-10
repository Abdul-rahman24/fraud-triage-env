from openenv.core.env_server.types import Action, Observation
from pydantic import Field

class FraudTriageAction(Action):
    decision: str = Field(
        ..., description="The triage decision. Approve (safe), Reject (fraud), Flag (uncertain case)."
    )
    reasoning: str = Field(
        default="No reasoning provided.", description="Explanation for the decision."
    )
    confidence_score: int = Field(
        default=100, ge=1, le=100, description="Confidence level of the decision from 1 to 100."
    )

class FraudTriageObservation(Observation):
    transaction_id: str = Field(default="", description="Unique ID")
    amount: float = Field(default=0.0, description="Amount in USD")
    merchant_category: str = Field(default="", description="Category of merchant")
    credit_score: int = Field(default=0, description="User credit score")
    has_chargebacks: bool = Field(default=False, description="History of chargebacks")