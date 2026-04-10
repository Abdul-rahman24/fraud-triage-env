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

# --- IDEA 2: RAG-LITE MOCK DATABASE ---
MOCK_DATABASE = {
    "Crypto": "- Case 101: $5000 Crypto, Credit 450, Chargebacks=True -> REJECT (High Risk Pattern)\n- Case 102: $100 Crypto, Credit 700, Chargebacks=False -> FLAG (Monitor)",
    "Groceries": "- Case 201: $30 Groceries, Credit 720, Chargebacks=False -> APPROVE (Low Risk, Normal Behavior)",
    "Electronics": "- Case 301: $800 Electronics, Credit 610, Chargebacks=True -> REJECT (Stolen Card Pattern)\n- Case 302: $400 Electronics, Credit 650, Chargebacks=False -> FLAG (Borderline)",
    "Travel": "- Case 401: $45 Travel, Credit 710, Chargebacks=False -> APPROVE (Low Risk)",
    "Retail": "- Case 501: $499 Retail, Credit 650, Chargebacks=True -> FLAG (Requires manual review due to chargebacks)"
}

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
                
                # STRICT [START] FORMAT
                print(f"[START] task={task} env=fraud_triage_env model={MODEL_NAME}", flush=True)
                
                steps = 0
                total_score = 0.0
                rewards_list = []
                
                try:
                    result = env.reset(task_id=task)
                    done = False
                    transaction_memory = [] # OLD IDEA 2: Short-Term Memory

                    while not done:
                        obs = result.observation
                        
                        # Memory Injection
                        memory_str = "\n".join(transaction_memory) if transaction_memory else "None (First transaction of session)"
                        
                        # RAG Injection
                        rag_examples = MOCK_DATABASE.get(obs.merchant_category, "No exact historical matches. Rely on standard rules.")
                        
                        transaction_data = f"""
                        - Transaction ID: {obs.transaction_id}
                        - Amount: ${obs.amount}
                        - Merchant Category: {obs.merchant_category}
                        - Credit Score: {obs.credit_score}
                        - Has Previous Chargebacks: {obs.has_chargebacks}
                        """

                        # ==========================================
                        # AGENT 1: THE MAKER (WITH CHAIN OF THOUGHT)
                        # ==========================================
                        maker_prompt = f"""
                        You are a Junior Fraud Analyst. Make an initial triage assessment based on the data.
                        
                        1. Recent Session Context (Memory):
                        {memory_str}
                        
                        2. Historical Reference Cases for {obs.merchant_category}:
                        {rag_examples}
                        
                        3. Current Transaction Data:
                        {transaction_data}
                        
                        Reasoning Framework (Think step-by-step):
                        - Step 1: Evaluate Amount vs. Category.
                        - Step 2: Evaluate Trust (Credit Score).
                        - Step 3: Evaluate History (Chargebacks and Session Memory).
                        
                        Provide a 'decision' (Approve/Reject/Flag), your step-by-step 'reasoning', and your 'confidence_score' (1-100).
                        """

                        maker_completion = llm_client.beta.chat.completions.parse(
                            model=MODEL_NAME,
                            messages=[
                                {"role": "system", "content": "You are a precise financial API."},
                                {"role": "user", "content": maker_prompt}
                            ],
                            response_format=FraudTriageAction,
                        )
                        maker_action = maker_completion.choices[0].message.parsed

                        # ==========================================
                        # AGENT 2: THE CHECKER
                        # ==========================================
                        checker_prompt = f"""
                        You are a Senior Fraud Auditor. Review the Junior Analyst's assessment. Correct any flaws in their logic.
                        
                        Current Transaction Data:
                        {transaction_data}
                        
                        Junior Analyst's Assessment:
                        - Proposed Decision: {maker_action.decision}
                        - Confidence: {maker_action.confidence_score}%
                        - Reasoning: {maker_action.reasoning}
                        
                        Critique their logic. Do you agree? Output the FINAL, authoritative decision (Approve/Reject/Flag), an updated reasoning, and your final confidence_score (1-100).
                        """

                        checker_completion = llm_client.beta.chat.completions.parse(
                            model=MODEL_NAME,
                            messages=[
                                {"role": "system", "content": "You are a precise financial API."},
                                {"role": "user", "content": checker_prompt}
                            ],
                            response_format=FraudTriageAction,
                        )
                        final_action = checker_completion.choices[0].message.parsed

                        # ==========================================
                        # THE GUARDRAIL
                        # ==========================================
                        if final_action.confidence_score < 75 and final_action.decision != "Flag":
                            original_decision = final_action.decision
                            final_action.decision = "Flag"
                            final_action.reasoning = f"[GUARDRAIL TRIGGERED] Overrode '{original_decision}' to 'Flag' due to low confidence ({final_action.confidence_score}%). " + final_action.reasoning

                        # Step environment
                        result = env.step(final_action)
                        
                        # Update Memory for next loop
                        transaction_memory.append(f"Category: {obs.merchant_category}, Amount: ${obs.amount}, Decision: {final_action.decision}")
                        if len(transaction_memory) > 3:
                            transaction_memory.pop(0)
                        
                        # Clamp and log score safely
                        reward = float(result.reward or 0.0)
                        clamped_reward = max(0.01, min(0.99, reward))
                        total_score += clamped_reward
                        rewards_list.append(f"{clamped_reward:.2f}")
                        
                        steps += 1
                        done = result.done
                        
                        # STRICT [STEP] FORMAT
                        done_str = "true" if done else "false"
                        action_str = final_action.decision.replace(" ", "_")
                        print(f"[STEP] step={steps} action={action_str} reward={clamped_reward:.2f} done={done_str} error=null", flush=True)

                    # STRICT [END] FORMAT
                    final_score = max(0.01, min(0.99, total_score))
                    success_str = "true" if final_score > 0.5 else "false"
                    rewards_str = ",".join(rewards_list)
                    
                    print(f"[END] success={success_str} steps={steps} score={final_score:.2f} rewards={rewards_str}", flush=True)
                    
                except Exception as e:
                    # Fail-safe emission
                    print(f"[END] success=false steps={steps} score=0.01 rewards=0.01", flush=True)
                    
    except Exception as env_error:
        pass

if __name__ == "__main__":
    run_baseline()