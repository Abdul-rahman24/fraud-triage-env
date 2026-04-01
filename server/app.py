"""
FastAPI application for the Fraud Triage Env Environment.
"""

from openenv.core.env_server.http_server import create_app

# Clean, standard absolute imports (works locally and in Docker)
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
    Entry point for direct execution.
    """
    import uvicorn
    uvicorn.run(app, host=host, port=port)

# USING SINGLE QUOTES FOR THE TEXT SCANNER
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    main(port=args.port)