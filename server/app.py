"""
FastAPI application for the Fraud Triage Env Environment.
"""

from openenv.core.env_server.http_server import create_app

# The original OpenEnv import structure that handles the uv run packaging!
try:
    from ..models import FraudTriageAction, FraudTriageObservation
    from .fraud_triage_env_environment import FraudTriageEnvironment
except (ModuleNotFoundError, ImportError):  # <-- WE ADDED IMPORTERROR HERE!
    from models import FraudTriageAction, FraudTriageObservation
    from server.fraud_triage_env_environment import FraudTriageEnvironment

app = create_app(
    FraudTriageEnvironment,
    FraudTriageAction,
    FraudTriageObservation,
    env_name="fraud_triage_env",
    max_concurrent_envs=1,
)

def main(host: str = "0.0.0.0", port: int = 8000):
    import uvicorn
    uvicorn.run(app, host=host, port=port)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    main(port=args.port)