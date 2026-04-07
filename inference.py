import os
import time
import urllib.request
import urllib.error
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

def get_env_url(max_retries=15, delay=2):
    """
    PROFESSIONAL FIX: Auto-Discovery Scanner
    Searches standard Docker hostnames to find the server.
    """
    print("🔍 Searching for live environment server...", flush=True)
    urls_to_try = [
        os.environ.get("ENV_BASE_URL"),
        os.environ.get("OPENENV_BASE_URL"),
        "http://server:8000",          
        "http://environment:8000",     
        "http://app:8000",             
        "http://localhost:8000",       
        "http://0.0.0.0:8000",
        "https://abdulrahman24-fraud-triage-env.hf.space" 
    ]
    
    for attempt in range(max_retries):
        for url in urls_to_try:
            if not url: continue
            url = url.rstrip('/')
            try:
                req = urllib.request.Request(f"{url}/schema", method="GET")
                with urllib.request.urlopen(req, timeout=2) as response:
                    print(f"✅ Success! Connected to {url}", flush=True)
                    return url
            except urllib.error.HTTPError as e:
                print(f"✅ Success! Connected to {url} (HTTP {e.code})", flush=True)
                return url
            except Exception:
                pass
        
        print(f"⏳ Server not ready yet. Retrying in {delay} seconds... (Attempt {attempt+1}/{max_retries})", flush=True)
        time.sleep(delay)
        
    print("❌ CRITICAL: Could not find live server. Defaulting to localhost.", flush=True)
    return "http://localhost:8000"


def run_baseline():
    if not API_KEY:
        print("❌ Error: API Key is missing. Set HF_TOKEN or OPENAI_API_KEY.", flush=True)
        return

    # 1. FIND THE SERVER URL DYNAMICALLY
    ENV_URL = get_env_url()

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

    print("=== INFERENCE PIPELINE INITIATED ===", flush=True)
    
    try:
        with FraudTriageEnv(base_url=ENV_URL).sync() as env:
            for task in tasks:
                
                # ==========================================
                # STRICT FORMATTING: [START] block
                # ==========================================
                print(f"[START] task={task}", flush=True)
                
                try:
                    result = env.reset(task_id=task)
                    done = False
                    total_score = 0.0
                    steps = 0
                    
                    # INNOVATION: AGENT MEMORY
                    transaction_memory = []

                    while not done:
                        obs = result.observation
                        
                        memory_str = "\n".join(transaction_memory) if transaction_memory else "None (First transaction of this session)"
                        
                        # INNOVATION: CHAIN OF THOUGHT PROMPTING
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
                        
                        # Store memory for the next loop
                        transaction_memory.append(f"Prev - Category: {obs.merchant_category}, Amount: ${obs.amount}, Decision: {action.decision}")
                        if len(transaction_memory) > 3:
                            transaction_memory.pop(0)
                        
                        # ==========================================
                        # DOUBLE CLAMP FIX: Clamp the step reward
                        # ==========================================
                        raw_reward = float(result.reward or 0.0)
                        clamped_reward = max(0.01, min(0.99, raw_reward))
                        
                        total_score += clamped_reward
                        steps += 1
                        done = result.done
                        
                        # ==========================================
                        # STRICT FORMATTING: [STEP] block
                        # ==========================================
                        print(f"[STEP] step={steps} reward={clamped_reward:.2f}", flush=True)
                        print(f"    Action: {action.decision} | Reason: {action.reasoning}", flush=True)

                    # ==========================================
                    # DOUBLE CLAMP FIX: Clamp the final score
                    # ==========================================
                    raw_score = total_score / steps if steps > 0 else 0.0
                    final_score = max(0.01, min(0.99, raw_score))
                    
                    print(f"[END] task={task} score={final_score:.2f} steps={steps}", flush=True)
                    
                except Exception as e:
                    print(f"❌ Error during task {task}: {str(e)}", flush=True)
                    
    except Exception as env_error:
        print(f"❌ CRITICAL ENVIRONMENT ERROR: {str(env_error)}", flush=True)

    print("=== INFERENCE PIPELINE COMPLETED ===", flush=True)

if __name__ == "__main__":
    run_baseline()