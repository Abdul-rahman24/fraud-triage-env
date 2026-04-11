import os
import time
import json
import urllib.request
import urllib.error
from typing import List, Optional
from openai import OpenAI
from dotenv import load_dotenv

from client import FraudTriageEnv
from models import FraudTriageAction

load_dotenv()

# ==========================================
# STRICT HACKATHON MANDATORY VARIABLES
# ==========================================
API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o-mini")
HF_TOKEN = os.getenv("HF_TOKEN")

if HF_TOKEN is None:
    raise ValueError("HF_TOKEN environment variable is required")

BENCHMARK_NAME = "fraud_triage_env"

# --- RAG-LITE MOCK DATABASE ---
MOCK_DATABASE = {
    "Crypto": "- Case 101: $5000 Crypto, Credit 450, Chargebacks=True -> REJECT (High Risk Pattern)\n- Case 102: $100 Crypto, Credit 700, Chargebacks=False -> FLAG (Monitor)",
    "Groceries": "- Case 201: $30 Groceries, Credit 720, Chargebacks=False -> APPROVE (Low Risk, Normal Behavior)",
    "Electronics": "- Case 301: $800 Electronics, Credit 610, Chargebacks=True -> REJECT (Stolen Card Pattern)\n- Case 302: $400 Electronics, Credit 650, Chargebacks=False -> FLAG (Borderline)",
    "Travel": "- Case 401: $45 Travel, Credit 710, Chargebacks=False -> APPROVE (Low Risk)",
    "Retail": "- Case 501: $499 Retail, Credit 650, Chargebacks=True -> FLAG (Requires manual review due to chargebacks)"
}

# ==========================================
# STRICT LOGGING FUNCTIONS (Matching Sample Exactly)
# ==========================================
def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)

def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )

# FIX: Re-added the 'score' parameter to match the parser's regex expectations
def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}", flush=True)


def get_env_url(max_retries=15, delay=2):
    urls_to_try = [
        os.getenv("ENV_BASE_URL"),
        os.getenv("OPENENV_BASE_URL"),
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
    ENV_URL = get_env_url()

    client = OpenAI(
        api_key=HF_TOKEN,
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
                
                log_start(task=task, env=BENCHMARK_NAME, model=MODEL_NAME)
                
                steps_taken = 0
                rewards: List[float] = []
                success = False
                
                try:
                    result = env.reset(task_id=task)
                    done = False
                    transaction_memory = [] 

                    while not done:
                        obs = result.observation
                        
                        memory_str = "\n".join(transaction_memory) if transaction_memory else "None (First transaction of session)"
                        rag_examples = MOCK_DATABASE.get(obs.merchant_category, "No exact historical matches. Rely on standard rules.")
                        
                        transaction_data = f"""
                        - Transaction ID: {obs.transaction_id}
                        - Amount: ${obs.amount}
                        - Merchant Category: {obs.merchant_category}
                        - Credit Score: {obs.credit_score}
                        - Has Previous Chargebacks: {obs.has_chargebacks}
                        """

                        system_prompt = "You are a precise financial API. You MUST output valid JSON with exactly three keys: 'decision' (must be 'Approve', 'Reject', or 'Flag'), 'reasoning' (string), and 'confidence_score' (integer 1-100)."
                        
                        action_decision = "Flag" # Safe default
                        action_reasoning = "Default Fallback"
                        
                        try:
                            # AGENT 1: THE MAKER
                            maker_prompt = f"Make an initial triage assessment based on the data.\nMemory: {memory_str}\nCases: {rag_examples}\nData: {transaction_data}\nOutput JSON."
                            maker_completion = client.chat.completions.create(
                                model=MODEL_NAME,
                                messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": maker_prompt}],
                                response_format={"type": "json_object"},
                            )
                            maker_data = json.loads(maker_completion.choices[0].message.content)

                            # AGENT 2: THE CHECKER
                            checker_prompt = f"Review the Junior Analyst's assessment.\nData: {transaction_data}\nProposed: {maker_data.get('decision')} | Confidence: {maker_data.get('confidence_score')}%\nCritique and output FINAL JSON decision."
                            checker_completion = client.chat.completions.create(
                                model=MODEL_NAME,
                                messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": checker_prompt}],
                                response_format={"type": "json_object"},
                            )
                            final_data = json.loads(checker_completion.choices[0].message.content)
                            
                            action_decision = str(final_data.get("decision", "Flag"))
                            action_reasoning = str(final_data.get("reasoning", "Parsed from JSON"))
                            confidence = int(final_data.get("confidence_score", 50))

                            # THE GUARDRAIL
                            if confidence < 75 and action_decision != "Flag":
                                action_reasoning = f"[GUARDRAIL TRIGGERED] Overrode '{action_decision}' to 'Flag' due to low confidence ({confidence}%). " + action_reasoning
                                action_decision = "Flag"
                                
                        except Exception as exc:
                            # FALLBACK FIX: Do not crash if LLM fails!
                            action_decision = "Flag"
                            action_reasoning = f"LLM Error Fallback: {str(exc)}"

                        # Load into Pydantic model and step environment
                        final_action = FraudTriageAction(
                            decision=action_decision,
                            reasoning=action_reasoning,
                            confidence_score=50
                        )

                        result = env.step(final_action)
                        
                        # Update Memory
                        transaction_memory.append(f"Category: {obs.merchant_category}, Amount: ${obs.amount}, Decision: {action_decision}")
                        if len(transaction_memory) > 3:
                            transaction_memory.pop(0)
                        
                        reward = float(result.reward or 0.0)
                        clamped_reward = max(0.01, min(0.99, reward))
                        rewards.append(clamped_reward)
                        
                        steps_taken += 1
                        done = result.done
                        error_msg = None
                        
                        log_step(step=steps_taken, action=action_decision.replace(" ", "_"), reward=clamped_reward, done=done, error=error_msg)

                    # FIX: Safely calculate score and pass it to log_end
                    raw_score = sum(rewards) / len(rewards) if rewards else 0.01
                    final_score = max(0.01, min(0.99, raw_score)) # STRICTLY bound between 0 and 1
                    success = final_score > 0.5
                    
                    log_end(success=success, steps=steps_taken, score=final_score, rewards=rewards)
                    
                except Exception as e:
                    # Environment crashed, log standard failure WITH SCORE
                    log_end(success=False, steps=steps_taken, score=0.01, rewards=[0.01])
                    
    except Exception as env_error:
        pass

if __name__ == "__main__":
    run_baseline()