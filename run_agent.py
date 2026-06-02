#!/usr/bin/env python3
"""
Minimal demo agent: iteratively optimizes model.py using an OpenAI-compatible LLM API.

Usage:
    export OPENAI_API_KEY=your-key
    export OPENAI_API_BASE=https://api.anthropic.com/v1/  # or openrouter, openai, etc.
    python run_agent.py --model claude-sonnet-4-6 --iterations 10 --mode dev

The agent loop:
    1. Read task.md + current model.py + eval history
    2. Ask LLM to produce an improved model.py
    3. Write it, run evaluator
    4. If score improved, keep it; otherwise revert
    5. Repeat
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import time

def call_llm(api_base, api_key, model, messages, max_tokens=16384, temperature=0.7):
    """Call OpenAI-compatible chat API."""
    import urllib.request
    url = f"{api_base.rstrip('/')}/chat/completions"
    body = json.dumps({
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }).encode()
    req = urllib.request.Request(url, data=body, headers={
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    })
    with urllib.request.urlopen(req, timeout=120) as resp:
        data = json.loads(resp.read())
    return data["choices"][0]["message"]["content"]


def run_evaluator(mode="sample", solution_dir="solution_template", require_neural=False):
    """Run evaluator and return parsed JSON result."""
    cmd = [sys.executable, "evaluator.py", "--mode", mode,
           "--solution-dir", solution_dir, "--budget", "0"]
    if require_neural:
        cmd.append("--require-neural")
    result = subprocess.run(cmd, capture_output=True, timeout=1800)
    try:
        return json.loads(result.stdout.decode())
    except:
        return {"feasible": False, "score": 0, "errors": [result.stderr.decode()[-500:]]}


def extract_code(response):
    """Extract Python code from LLM response (between ```python and ```)."""
    if "```python" in response:
        code = response.split("```python")[1].split("```")[0]
        return code.strip()
    elif "```" in response:
        code = response.split("```")[1].split("```")[0]
        return code.strip()
    return response.strip()


def main():
    parser = argparse.ArgumentParser(description="Demo optimization agent")
    parser.add_argument("--model", default="claude-sonnet-4-6")
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--mode", default="dev", help="Evaluation mode")
    parser.add_argument("--require-neural", action="store_true")
    parser.add_argument("--solution-dir", default="solution_template")
    args = parser.parse_args()

    api_key = os.environ.get("OPENAI_API_KEY", "")
    api_base = os.environ.get("OPENAI_API_BASE", "https://api.openai.com/v1")
    if not api_key:
        print("Error: set OPENAI_API_KEY environment variable")
        sys.exit(1)

    model_path = os.path.join(args.solution_dir, "model.py")
    task_md = open("task.md").read()

    # Initial evaluation
    print(f"[0/{args.iterations}] Evaluating baseline...")
    result = run_evaluator(args.mode, args.solution_dir, args.require_neural)
    best_score = result["score"] if result["feasible"] else 0
    best_code = open(model_path).read()
    history = [{"iteration": 0, "score": best_score, "feasible": result["feasible"]}]
    print(f"  Baseline score: {best_score:.4f}")

    for i in range(1, args.iterations + 1):
        current_code = open(model_path).read()

        # Build prompt with task, current code, and history
        history_str = "\n".join(
            f"  Iter {h['iteration']}: score={h['score']:.4f} feasible={h['feasible']}"
            for h in history[-10:]
        )

        messages = [
            {"role": "system", "content": "You are optimizing a text compression model. "
             "Output ONLY the complete improved model.py code inside ```python``` blocks. "
             "No explanations before the code."},
            {"role": "user", "content": f"""## Task
{task_md[:2000]}

## Current model.py (score: {best_score:.4f})
```python
{current_code}
```

## Recent history
{history_str}

## Feedback from last evaluation
{json.dumps(result.get('per_chunk', [])[:1], indent=2)[:1000]}

Improve this model to get a higher compression score. Output the complete improved model.py."""}
        ]

        print(f"\n[{i}/{args.iterations}] Calling LLM...")
        try:
            response = call_llm(api_base, api_key, args.model, messages)
            new_code = extract_code(response)
        except Exception as e:
            print(f"  LLM error: {e}")
            history.append({"iteration": i, "score": 0, "feasible": False, "error": str(e)})
            continue

        # Write and evaluate
        backup = open(model_path).read()
        with open(model_path, "w") as f:
            f.write(new_code)

        print(f"  Evaluating ({args.mode})...")
        result = run_evaluator(args.mode, args.solution_dir, args.require_neural)
        score = result["score"] if result["feasible"] else 0
        feasible = result["feasible"]

        if feasible and score > best_score:
            best_score = score
            best_code = new_code
            print(f"  NEW BEST: {score:.4f} (+{score - history[-1].get('score', 0):.4f})")
        elif feasible:
            print(f"  Score: {score:.4f} (no improvement, reverting)")
            with open(model_path, "w") as f:
                f.write(best_code)
        else:
            errors = result.get("errors", [])[:1]
            print(f"  INFEASIBLE: {errors[0][:100] if errors else 'unknown'}")
            with open(model_path, "w") as f:
                f.write(best_code)

        history.append({"iteration": i, "score": score, "feasible": feasible})

    # Save trajectory
    traj_path = f"trajectory_{args.model.replace('/', '_')}_{int(time.time())}.jsonl"
    with open(traj_path, "w") as f:
        for h in history:
            f.write(json.dumps(h) + "\n")
    print(f"\nDone. Best score: {best_score:.4f}")
    print(f"Trajectory saved to {traj_path}")


if __name__ == "__main__":
    main()
