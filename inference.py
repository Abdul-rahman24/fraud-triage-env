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
                    return url
            except urllib.error.HTTPError as e:
                return url
            except Exception:
                pass
        time.sleep(delay)
        
    return "http://localhost:8000"


def run_baseline():
    if not API_KEY:
        print("Error: API Key is missing.", flush=True)
        return

    ENV_URL = get_env_url()

    llm_client = OpenAI(
        api_key=API_KEY,
        base_url=API_BASE_URL
    )
    
    tasks = [
        "easy_fraud_detection",
        "medium_fraud_detection",
        "hard_fraud_detection"
    ]

    try:
        with FraudTriageEnv(base_url=ENV_URL).sync() as env:
            for task in tasks:
                
                # ==========================================
                # 1. STRICT [START] FORMAT
                # ==========================================
                print(f"[START] task={task} env=fraud_triage_env model={MODEL_NAME}", flush=True)
                
                steps = 0
                total_score = 0.0
                rewards_list = []
                
                try:
                    result = env.reset(task_id=task)
                    done = False
                    transaction_memory = []

                    while not done:
                        obs = result.observation
                        memory_str = "\n".join(transaction_memory) if transaction_memory else "None"
                        
                        prompt = f"""
                        You are a senior fraud detection analyst. 
                        Analyze the following transaction data and make a strict triage decision.
                        
                        Recent Context: {memory_str}
                        
                        Transaction ID: {obs.transaction_id}
                        Amount: ${obs.amount}
                        Merchant Category: {obs.merchant_category}
                        Credit Score: {obs.credit_score}
                        Has Previous Chargebacks: {obs.has_chargebacks}
                        
                        Rules:
                        - 'Approve': Safe, normal transaction.
                        - 'Reject': High probability of fraud.
                        - 'Flag': Uncertain, requires human review.
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
                        
                        transaction_memory.append(f"Category: {obs.merchant_category}, Amount: ${obs.amount}, Decision: {action.decision}")
                        if len(transaction_memory) > 3:
                            transaction_memory.pop(0)
                        
                        # Keep reward strictly bounded per guidelines
                        reward = float(result.reward or 0.0)
                        clamped_reward = max(0.01, min(0.99, reward))
                        total_score += clamped_reward
                        rewards_list.append(f"{clamped_reward:.2f}")
                        
                        steps += 1
                        done = result.done
                        
                        # Format booleans as lowercase strings
                        done_str = "true" if done else "false"
                        action_str = action.decision.replace(" ", "_")
                        
                        # ==========================================
                        # 2. STRICT [STEP] FORMAT (Single line, no newlines)
                        # ==========================================
                        print(f"[STEP] step={steps} action={action_str} reward={clamped_reward:.2f} done={done_str} error=null", flush=True)

                    # ==========================================
                    # 3. STRICT [END] FORMAT
                    # ==========================================
                    final_score = max(0.01, min(0.99, total_score))
                    success_str = "true" if final_score > 0.5 else "false"
                    rewards_str = ",".join(rewards_list)
                    
                    print(f"[END] success={success_str} steps={steps} score={final_score:.2f} rewards={rewards_str}", flush=True)
                    
                except Exception as e:
                    # MUST emit [END] even on exception per the rules
                    print(f"[END] success=false steps={steps} score=0.01 rewards=0.01", flush=True)
                    
    except Exception as env_error:
        pass

if __name__ == "__main__":
    run_baseline()