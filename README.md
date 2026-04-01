---
title: AI Fraud Triage Environment
emoji: ⏲️
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
app_port: 7860
base_path: /web
tags:
  - openenv
  - finance
  - agent-eval
---

# AI Financial Fraud Triage Environment

## Environment Description & Motivation
Financial institutions process millions of transactions daily. While standard rule-based systems catch obvious fraud, they generate thousands of "gray-area" alerts that require expensive human manual review (Triage). 

**Motivation:** This OpenEnv environment evaluates an LLM agent's capacity to act as a **Senior Fraud Detection Analyst**. By accurately triaging alerts into Approve, Reject, or Flag categories based on conflicting signals, highly capable LLMs can drastically reduce the human operational burden in modern banking systems.

## Action & Observation Spaces

### Observation Space (The Alert)
The environment feeds the agent simulated transaction data via the strictly typed `FraudTriageObservation` Pydantic model:
- `transaction_id` (str): Unique identifier.
- `amount` (float): Transaction amount in USD.
- `merchant_category` (str): Sector of purchase (e.g., Crypto, Groceries, Retail).
- `credit_score` (int): Customer's credit score (300-850).
- `has_chargebacks` (bool): Boolean indicating historical chargeback disputes.

### Action Space (The Agent's Decision)
The agent must respond with a strictly typed `FraudTriageAction`:
- `decision` (Literal): Must be exactly:
  - `Approve`: Safe, normal transaction.
  - `Reject`: High probability of fraud.
  - `Flag`: Uncertain, requires human review.
- `reasoning` (str): The agent's logical justification for the decision.

## Tasks & Expected Difficulty

The environment features 3 tasks with varying complexities:
1. **Easy Fraud Detection (`easy_fraud_detection`):** Clear signals. E.g., Massive amounts on Crypto with a terrible credit score vs. tiny grocery bills with an excellent score.
2. **Medium Fraud Detection (`medium_fraud_detection`):** Mixed signals. E.g., Moderate amounts in higher-risk categories (Electronics) but with fair credit scores, requiring the agent to utilize the `Flag` action safely.
3. **Hard Fraud Detection (`hard_fraud_detection`):** Conflicting signals. E.g., Perfect credit scores but a history of chargebacks, requiring deep reasoning on risk vs. reward.

**Grading Logic:**
- **+1.0:** Correctly identifying the ground truth.
- **+0.4:** Partial progress reward for choosing `Flag` on a difficult fraudulent transaction, mimicking real-world risk mitigation.
- **0.0:** Incorrect evaluation.

## Setup & Usage Instructions

### Prerequisites
- Python 3.10+
- The `uv` package manager

### Running Locally
1. Clone the repository and install dependencies using `uv`.
2. Ensure you have defined your local environment variables in a `.env` file or exported them:
   - `HF_TOKEN` (or `OPENAI_API_KEY`)
   - `API_BASE_URL`
   - `MODEL_NAME`

3. **Start the Environment Server:**
```bash
uv run server