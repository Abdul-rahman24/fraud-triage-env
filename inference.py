import os
import time
import socket
from urllib.parse import urlparse
from openai import OpenAI
from dotenv import load_dotenv

from client import FraudTriageEnv
from models import FraudTriageAction

load_dotenv()

# ==========================================
# HACKATHON MANDATORY VARIABLES
# ==========================================
API_BASE_URL = os.environ.get("API_BASE_URL", "https://generativelanguage.googleapis.com/v1beta/openai/")
API_KEY = os.environ.get("HF_TOKEN") or os.environ.get("OPENAI_API_KEY")
MODEL_NAME = os.environ.get("MODEL_NAME", "gemini-2.5-pro")

# The URL where the environment server lives
ENV_URL = os.environ.get("ENV_BASE_URL", "http://127.0.0.1:8000")

def wait_for_server(url, max_retries=15, delay=2):
    """
    PROFESSIONAL FIX: Raw TCP Socket Ping.
    This safely waits for the server port to open BEFORE initializing the async client.
    This prevents the unhandled asyncio exceptions from crashing Phase 2.
    """
    parsed = urlparse(url)
    host = parsed.hostname or "127.0.0.1"
    port = parsed.port or 8000
    
    print(f"⏳ Waiting for server to open at {host}:{port}...")
    for attempt in range(max_retries):
        try:
            # Try to establish a raw TCP connection
            with socket.create_connection((host, port), timeout=2):
                print(f"✅ Server is live on attempt {attempt + 1}!")
                return
        except OSError:
            print(f"   Port closed, waiting... ({attempt + 1}/{max_retries})")
            time.sleep(delay)
            
    raise Exception(f"❌ CRITICAL ERROR: Server at {url} never booted up.")


def run_baseline():
    if not API_KEY:
        print("❌ Error: API Key is missing. Set HF_TOKEN or OPENAI_API_KEY.")
        return

    # 1. GUARANTEE THE SERVER IS READY FIRST
    wait_for_server(ENV_URL)

    # 2. INITIALIZE CLIENT
    llm_client = OpenAI(
        api_key=API_KEY,
        base_url=API_BASE_URL
    )
    
    tasks = [
        "easy_fraud_detection",
        "medium_fraud_detection",
        "hard_fraud_detection"
    ]

    print("START")
    
    # Now that the port is confirmed open, this will connect flawlessly.
    with FraudTriageEnv(base_url=ENV_URL).sync() as env:
        for task in tasks:
            print(f"\n{'-'*50}")
            print(f"Evaluating Task: {task}")
            
            try:
                result = env.reset(task_id=task)
                done = False
                total_score = 0.0
                steps = 0
                
                # ==========================================
                # INNOVATION: AGENT MEMORY
                # ==========================================
                transaction_memory = []

                while not done:
                    obs = result.observation
                    
                    # Format recent history for the prompt
                    memory_str = "\n".join(transaction_memory) if transaction_memory else "None (First transaction of this session)"
                    
                    # ==========================================
                    # INNOVATION: CHAIN OF THOUGHT PROMPTING
                    # ==========================================
                    prompt = f"""
                    You are a senior fraud detection analyst. 
                    Analyze the following transaction data and make a strict triage decision.
                    
                    Recent Transaction Context (Memory):
                    {memory_str}
                    
                    Current Transaction Data:
                    - Transaction ID: {obs.transaction_id}
                    - Amount: ${obs.amount}
                    - Merchant Category: {obs.merchant_category}
                    - Credit Score: {obs.credit_score}
                    - Has Previous Chargebacks: {obs.has_chargebacks}
                    
                    Reasoning Framework (Think step-by-step):
                    1. Evaluate Amount vs. Category: Is this an unusually high amount for this merchant type? Are there patterns from the recent history?
                    2. Evaluate Trust: Does the credit score suggest financial stability?
                    3. Evaluate History: Are there previous chargebacks indicating a pattern of disputes or stolen cards?
                    
                    Rules:
                    - 'Approve': Safe, normal transaction (High credit, low amount, no chargebacks).
                    - 'Reject': High probability of fraud (High amount, low credit, previous chargebacks).
                    - 'Flag': Uncertain, requires human review (Conflicting signals, e.g., high credit but previous chargebacks).
                    """

                    completion = llm_client.beta.chat.completions.parse(
                        model=MODEL_NAME,
                        messages=[
                            {"role": "system", "content": "You are a precise financial API. Always adhere strictly to the schema."},
                            {"role": "user", "content": prompt}
                        ],
                        response_format=FraudTriageAction,
                    )

                    action = completion.choices[0].message.parsed
                    result = env.step(action)
                    
                    # Store memory for the next loop (keep only the last 3)
                    transaction_memory.append(f"Prev - Category: {obs.merchant_category}, Amount: ${obs.amount}, Decision: {action.decision}")
                    if len(transaction_memory) > 3:
                        transaction_memory.pop(0)
                    
                    reward = float(result.reward or 0.0)
                    total_score += reward
                    steps += 1
                    done = result.done
                    
                    feedback = result.observation.metadata.get('feedback', 'No feedback provided.')
                    
                    print(f"STEP: {action.decision} | Score: {result.reward} | Reason: {action.reasoning}")
                    print(f"          Reasoning: {action.reasoning}")
                    print(f"          Feedback: {feedback}\n")

                final_score = total_score / steps if steps > 0 else 0
                print(f"✅ Finished {task} | Final Score: {final_score:.2f} / 1.00")
                
            except Exception as e:
                print(f"❌ Error during task {task}: {str(e)}")

    print(f"{'-'*50}")
    print("END")

if __name__ == "__main__":
    run_baseline()