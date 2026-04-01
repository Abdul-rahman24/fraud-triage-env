"""
FastAPI application for the Fraud Triage Env Environment.
"""

try:
    from openenv.core.env_server.http_server import create_app
except Exception as e:
    raise ImportError(
        "openenv is required for the web interface. Install dependencies with 'uv sync'"
    ) from e

# Our robust import fix
try:
    from ..models import FraudTriageAction, FraudTriageObservation
    from .fraud_triage_env_environment import FraudTriageEnvironment
except (ModuleNotFoundError, ImportError):
    from models import FraudTriageAction, FraudTriageObservation
    from server.fraud_triage_env_environment import FraudTriageEnvironment

# Create the app
app = create_app(
    FraudTriageEnvironment,
    FraudTriageAction,
    FraudTriageObservation,
    env_name="fraud_triage_env",
    max_concurrent_envs=1, 
)

def main(host: str = "0.0.0.0", port: int = 8000):
    """
    Entry point for direct execution via uv run or python -m.
    """
    import uvicorn
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    main(port=args.port)