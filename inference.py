import os
import time
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

# The URL where the environment server lives (evaluators often inject this)
ENV_URL = os.environ.get("ENV_BASE_URL", "http://localhost:8000")

def run_baseline():
    if not API_KEY:
        print("❌ Error: API Key is missing. Set HF_TOKEN or OPENAI_API_KEY.")
        return

    # Initialize client using the mandatory variables
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
    
    # ==========================================
    # RETRY LOOP FIX FOR PHASE 2 RACE CONDITION
    # ==========================================
    max_retries = 10
    for attempt in range(max_retries):
        try:
            with FraudTriageEnv(base_url=ENV_URL).sync() as env:
                for task in tasks:
                    print(f"\n{'-'*50}")
                    print(f"Evaluating Task: {task}")
                    
                    try:
                        result = env.reset(task_id=task)
                        done = False
                        total_score = 0.0
                        steps = 0

                        while not done:
                            obs = result.observation
                            
                            prompt = f"""
                            You are a senior fraud detection analyst. 
                            Analyze the following transaction data and make a triage decision.
                            
                            Transaction Data:
                            - Transaction ID: {obs.transaction_id}
                            - Amount: ${obs.amount}
                            - Merchant Category: {obs.merchant_category}
                            - Credit Score: {obs.credit_score}
                            - Has Previous Chargebacks: {obs.has_chargebacks}
                            
                            Rules:
                            - 'Approve': Safe, normal transaction.
                            - 'Reject': High probability of fraud.
                            - 'Flag': Uncertain, requires human review.
                            """

                            # Pass the mandatory MODEL_NAME variable here
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
            
            # If we successfully completed the 'with' block, break out of the retry loop
            break 
            
        except Exception as e:
            if attempt < max_retries - 1:
                print(f"Server not ready, waiting... (Attempt {attempt+1}/{max_retries})")
                time.sleep(3)
            else:
                print(f"❌ Failed to connect after {max_retries} attempts. Error: {str(e)}")
                raise e # Crash loudly so the grader sees the error

    print(f"{'-'*50}")
    print("END")

if __name__ == "__main__":
    run_baseline()