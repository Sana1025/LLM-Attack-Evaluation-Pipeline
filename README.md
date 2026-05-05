# LLM Attack Evaluation Pipeline

A structured framework to evaluate how Large Language Models (LLMs) respond to **adversarial prompts**, measure **attack success rates**, and analyze **failure mechanisms**.

This pipeline helps researchers understand *where models break*, *why they fail*, and *how to improve defenses*.



## Overview

This project automates:

* Running adversarial prompts against an LLM
* Classifying responses (success / refusal / partial)
* Identifying failure mechanisms
* Logging results into **Excel + JSON**
* Generating attack success metrics (ASR)

It’s designed for **LLM safety research, red teaming, and robustness evaluation**.

## Project Structure

```
.
├── run_pipeline.py        # Main script (evaluation engine)
├── LLM_Attack_Dataset.xlsx  # Input prompts + output logging
├── results.json           # Exported results
```



## Attack Categories

The pipeline evaluates different adversarial strategies:

| Code | Category              | Description                   |
| ---- | --------------------- | ----------------------------- |
| RB   | Role-Based            | Persona manipulation          |
| IH   | Instruction Hierarchy | Override system instructions  |
| CI   | Context Injection     | Misleading contextual framing |
| MS   | Multi-step Reasoning  | Gradual intent escalation     |



## Quick Start

### 1. Install dependencies

```bash
pip install pandas openpyxl openai
```

### 2. Run the pipeline

```bash
python run_pipeline.py \
  --api-key YOUR_API_KEY
```

### Optional flags

```bash
--dry-run          # No API calls (testing mode)
--category RB      # Filter by category (RB, IH, CI, MS)
```



## How It Works

### 1. Prompt Execution

Prompts are loaded from Excel and sent to the LLM via API.

### 2. Response Classification

Each response is categorized using heuristic rules:

* **Attack Success** → Model gives full harmful response
* **Partial Success** → Model hedges but leaks information
* **Hard Refusal** → Model blocks the request

### 3. Failure Mechanism Detection

The system identifies *why* the model failed:

* Role identity override
* Chain blindness
* Context injection
* Instruction hierarchy confusion


## Output

### Excel (`LLM_Attack_Dataset.xlsx`)

* Structured results log
* Styled cells for readability
* Defense suggestions included

### JSON (`results.json`)

* Full raw outputs
* Timestamped results
* Easy for further analysis


## Defense Insights

The pipeline also suggests improvements based on failure patterns:

| Failure Mechanism     | Suggested Defense                    |
| --------------------- | ------------------------------------ |
| Role override         | Strengthen system prompt constraints |
| Chain blindness       | Add multi-step intent tracking       |
| Context injection     | Re-evaluate long-context safety      |
| Instruction confusion | Define instruction priority clearly  |



## Example Workflow

1. Add prompts to Excel dataset
2. Run pipeline
3. Review:
   * Attack success cases
   * Partial leaks
   * Failure reasons
4. Improve model defenses


## Notes

* This tool is intended **only for research and safety evaluation**
* Do not use it to generate or deploy harmful content
* Ensure compliance with API usage policies


## Future Improvements

* Multi-model comparison dashboard
* Visualization of attack trends
* Integration with open-source LLMs
* Advanced attack generation (auto red teaming)


## Inspiration

This project is built around the growing need for **robust, safe, and trustworthy AI systems**, especially as LLMs become more widely deployed.
