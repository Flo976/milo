"""
Milo Orchestrator — Autonomous self-healing pipeline runner.

Monitors training, launches evaluation, auto-fixes errors via Claude Code.
Logs everything including fixes for full traceability.

Usage:
    python3 orchestrator.py
    python3 orchestrator.py --skip-training
    python3 orchestrator.py --skip-baseline
"""
import argparse
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime


# === Config ===
MILO_DIR = "/home/florent/milo"
VENV_ACTIVATE = "source /home/florent/milo/venv/bin/activate"
SCRIPTS_DIR = os.path.join(MILO_DIR, "scripts")
MODELS_DIR = os.path.join(MILO_DIR, "models")
EVAL_DIR = os.path.join(MILO_DIR, "evaluation")
LOG_FILE = os.path.join(MILO_DIR, "orchestrator.log")
STATUS_FILE = os.path.join(MILO_DIR, "pipeline_status.json")
FIXES_LOG = os.path.join(MILO_DIR, "fixes_history.log")

TRAINING_LOG = os.path.join(MILO_DIR, "training.log")
MODEL_OUTPUT = os.path.join(MODELS_DIR, "whisper-mg-v1")
MODEL_FINAL = os.path.join(MODEL_OUTPUT, "final")

WER_THRESHOLD = 50.0
MAX_FIX_ATTEMPTS = 3


# === Logging ===

def log(msg):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = "[{}] {}".format(timestamp, msg)
    print(line, flush=True)
    with open(LOG_FILE, "a") as f:
        f.write(line + "\n")


def log_fix(step, error, claude_response, fix_applied):
    """Log a fix attempt for traceability."""
    entry = {
        "timestamp": datetime.now().isoformat(),
        "step": step,
        "error": error[:2000],
        "claude_response": claude_response[:5000],
        "fix_applied": fix_applied,
    }
    with open(FIXES_LOG, "a") as f:
        f.write("\n" + "=" * 70 + "\n")
        f.write("FIX ATTEMPT — {} — {}\n".format(entry["timestamp"], step))
        f.write("=" * 70 + "\n")
        f.write("\n--- ERROR ---\n")
        f.write(error[:2000] + "\n")
        f.write("\n--- CLAUDE RESPONSE ---\n")
        f.write(claude_response[:5000] + "\n")
        f.write("\n--- FIX APPLIED ---\n")
        f.write(str(fix_applied) + "\n")
        f.write("=" * 70 + "\n")

    log("Fix logged to {}".format(FIXES_LOG))


def update_status(step, status, details=None):
    try:
        with open(STATUS_FILE, "r") as f:
            data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        data = {"steps": {}, "started": datetime.now().isoformat()}

    data["steps"][step] = {
        "status": status,
        "updated": datetime.now().isoformat(),
    }
    if details:
        data["steps"][step]["details"] = details
    data["last_update"] = datetime.now().isoformat()

    with open(STATUS_FILE, "w") as f:
        json.dump(data, f, indent=2)


# === Shell helpers ===

def run_cmd(cmd, timeout=None):
    log("CMD: {}".format(cmd[:200]))
    try:
        result = subprocess.run(
            ["bash", "-c", cmd],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        return result.returncode, result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        return -1, "", "TIMEOUT"


# === Claude Code self-healing ===

def call_claude_fix(step_name, script_path, error_msg, context=""):
    """
    Call Claude Code in headless mode to diagnose and fix an error.
    Returns (success, response_text, fix_description).
    """
    log("AUTOFIX: Calling Claude Code for step '{}'".format(step_name))

    prompt = """Tu es l'orchestrateur du projet Milo (fine-tuning Whisper pour le malagasy).
Le script "{script}" a planté pendant l'etape "{step}".

=== ERREUR ===
{error}

=== CONTEXTE ===
- Projet: /home/florent/milo
- Venv: /home/florent/milo/venv
- GPU: RTX 5070 Ti, CUDA 13.1, PyTorch nightly cu128
- Dataset: badrex/malagasy-speech-full (28371 train, 3099 val, 3101 test)
- datasets==2.21 (pas torchcodec, utiliser soundfile)
{extra_context}

=== INSTRUCTIONS ===
1. Diagnostique l'erreur
2. Corrige le script {script} directement
3. Ne change QUE ce qui est necessaire pour fixer le bug
4. Reponds avec un résumé court de ce que tu as corrige

IMPORTANT: Corrige le fichier directement, ne fais pas que suggerer.""".format(
        script=script_path,
        step=step_name,
        error=error_msg[-3000:],
        extra_context=context,
    )

    cmd = 'claude -p "{}" --allowedTools "Edit,Read,Bash,Write" 2>&1'.format(
        prompt.replace('"', '\\"').replace('\n', '\\n')
    )

    # Use a temp file for the prompt to avoid shell escaping issues
    prompt_file = "/tmp/milo_fix_prompt.txt"
    with open(prompt_file, "w") as f:
        f.write(prompt)

    cmd = 'cat {} | /usr/bin/claude -p --allowedTools "Edit,Read,Bash,Write" 2>&1'.format(prompt_file)

    code, stdout, stderr = run_cmd(cmd, timeout=300)

    full_response = stdout + stderr
    success = code == 0

    log("AUTOFIX result: code={}, response_len={}".format(code, len(full_response)))

    if full_response:
        # Log first 500 chars of response
        log("AUTOFIX response: {}".format(full_response[:500]))

    return success, full_response, "Claude fix attempt (exit code {})".format(code)


def run_with_autofix(step_name, cmd, script_path, timeout=7200, max_retries=MAX_FIX_ATTEMPTS):
    """
    Run a command. If it fails, call Claude to fix it, then retry.
    Returns (success, stdout, stderr, num_fixes).
    """
    for attempt in range(max_retries + 1):
        label = "attempt {}/{}".format(attempt + 1, max_retries + 1)
        log("Running {} ({})".format(step_name, label))

        code, stdout, stderr = run_cmd(cmd, timeout=timeout)

        if code == 0:
            log("{} succeeded ({})".format(step_name, label))
            return True, stdout, stderr, attempt

        # Failed — extract error
        error_output = stderr[-3000:] if stderr else stdout[-3000:]
        log("{} FAILED ({}): {}".format(step_name, label, error_output[:300]))

        if attempt >= max_retries:
            log("Max retries reached for {}. Giving up.".format(step_name))
            return False, stdout, stderr, attempt

        # Call Claude to fix
        log("Attempting autofix {}/{}...".format(attempt + 1, max_retries))
        update_status(step_name, "fixing", {
            "attempt": attempt + 1,
            "error": error_output[:200]
        })

        fix_ok, claude_response, fix_desc = call_claude_fix(
            step_name, script_path, error_output
        )

        log_fix(step_name, error_output, claude_response, fix_desc)

        if not fix_ok:
            log("Claude fix call failed. Will retry anyway...")

        # Sync script back if it was on Windows mount
        win_script = "'/mnt/c/Users/Florent Didelot/Desktop/milo/scripts/{}'".format(
            os.path.basename(script_path)
        )
        run_cmd("cp {} {} 2>/dev/null; cp {} {} 2>/dev/null".format(
            script_path, win_script, win_script, script_path
        ))

        log("Retrying {}...".format(step_name))
        time.sleep(5)

    return False, "", "max retries", max_retries


# === Training monitoring ===

def check_training_running():
    code, out, _ = run_cmd("pgrep -af 'python.*03_train.py' | grep -v pgrep | grep -v tmux | head -1")
    return code == 0 and out.strip() != ""


def check_training_complete():
    if not os.path.exists(TRAINING_LOG):
        return False, None, ""

    with open(TRAINING_LOG, "r") as f:
        content = f.read()

    if "TRAINING COMPLETE" in content:
        return True, "success", ""

    if not check_training_running():
        # Process died — find the error
        lines = content.split("\n")
        # Find traceback
        error_lines = []
        in_traceback = False
        for line in lines:
            if "Traceback" in line:
                in_traceback = True
            if in_traceback:
                error_lines.append(line)
        error_msg = "\n".join(error_lines[-30:]) if error_lines else "Process died without traceback"
        return True, "error", error_msg

    return False, None, ""


def get_training_progress():
    if not os.path.exists(TRAINING_LOG):
        return "no log"

    with open(TRAINING_LOG, "r") as f:
        content = f.read()

    lines = content.split("\n")

    # Check for training steps (look for loss logging)
    step_pattern = re.compile(r"'loss':\s*([\d.]+).*'step':\s*(\d+)")
    step_matches = []
    for line in lines:
        m = step_pattern.search(line)
        if m:
            step_matches.append((float(m.group(1)), int(m.group(2))))

    if step_matches:
        loss, step = step_matches[-1]
        return "step {} | loss {:.4f}".format(step, loss)

    # Check for map progress
    map_pattern = re.compile(r"Map:.*?(\d+)%")
    map_matches = list(map_pattern.finditer(content))
    if map_matches:
        pct = map_matches[-1].group(1)
        return "preprocessing: {}%".format(pct)

    return "starting..."


def find_best_checkpoint():
    if os.path.exists(MODEL_FINAL):
        return MODEL_FINAL
    if not os.path.exists(MODEL_OUTPUT):
        return None
    checkpoints = sorted([
        d for d in os.listdir(MODEL_OUTPUT) if d.startswith("checkpoint-")
    ], key=lambda x: int(x.split("-")[1]) if x.split("-")[1].isdigit() else 0)
    if checkpoints:
        return os.path.join(MODEL_OUTPUT, checkpoints[-1])
    return None


# === Pipeline steps ===

def step_wait_training():
    log("=" * 50)
    log("STEP: Waiting for training")
    update_status("training", "in_progress")

    poll_interval = 30
    last_progress = ""

    while True:
        done, result, error_msg = check_training_complete()

        if done and result == "success":
            log("Training COMPLETE")
            update_status("training", "complete")
            return True

        if done and result == "error":
            log("Training FAILED: {}".format(error_msg[:300]))
            update_status("training", "failed", error_msg[:200])

            # Try autofix: fix the script, then relaunch training
            fix_ok, claude_resp, fix_desc = call_claude_fix(
                "training", "/home/florent/milo/scripts/03_train.py", error_msg
            )
            log_fix("training", error_msg, claude_resp, fix_desc)

            if fix_ok:
                log("Relaunching training after fix...")
                # Clear old log
                run_cmd("mv {} {}.bak".format(TRAINING_LOG, TRAINING_LOG))
                # Relaunch in tmux
                run_cmd(
                    "tmux kill-session -t milo 2>/dev/null; "
                    "tmux new-session -d -s milo '"
                    "{} && python3 /home/florent/milo/scripts/03_train.py "
                    "--epochs 10 --batch-size 4 --grad-accum 4 "
                    "--eval-steps 1000 --save-steps 1000 "
                    "2>&1 | tee {}'".format(VENV_ACTIVATE, TRAINING_LOG)
                )
                update_status("training", "relaunched_after_fix")
                time.sleep(30)
                continue
            else:
                log("Could not fix training. Stopping.")
                return False

        progress = get_training_progress()
        if progress != last_progress:
            log("Progress: {}".format(progress))
            last_progress = progress

        time.sleep(poll_interval)


def step_evaluate_baseline():
    log("=" * 50)
    log("STEP: Baseline evaluation")
    update_status("baseline", "in_progress")

    output_file = os.path.join(EVAL_DIR, "baseline_results.json")
    cmd = "{} && python3 {}/04_evaluate.py --model openai/whisper-medium --max-samples 200 --output {}".format(
        VENV_ACTIVATE, SCRIPTS_DIR, output_file
    )

    success, stdout, stderr, fixes = run_with_autofix(
        "baseline",
        cmd,
        os.path.join(SCRIPTS_DIR, "04_evaluate.py"),
        timeout=3600,
    )

    if not success:
        log("Baseline eval FAILED after {} fix attempts".format(fixes))
        update_status("baseline", "failed")
        return None

    log(stdout[-300:] if stdout else "")

    try:
        with open(output_file, "r") as f:
            results = json.load(f)
        wer_score = results["summary"]["wer"]
        log("Baseline WER: {:.1f}%".format(wer_score))
        update_status("baseline", "complete", {
            "wer": wer_score,
            "fixes_needed": fixes
        })
        return results["summary"]
    except Exception as e:
        log("Could not parse baseline results: {}".format(e))
        update_status("baseline", "complete")
        return None


def step_evaluate():
    log("=" * 50)
    log("STEP: Fine-tuned model evaluation")
    update_status("evaluation", "in_progress")

    model_path = find_best_checkpoint()
    if not model_path:
        log("ERROR: No model checkpoint found")
        update_status("evaluation", "failed", "no checkpoint")
        return None

    log("Model: {}".format(model_path))
    output_file = os.path.join(EVAL_DIR, "results.json")

    cmd = "{} && python3 {}/04_evaluate.py --model {} --output {}".format(
        VENV_ACTIVATE, SCRIPTS_DIR, model_path, output_file
    )

    success, stdout, stderr, fixes = run_with_autofix(
        "evaluation",
        cmd,
        os.path.join(SCRIPTS_DIR, "04_evaluate.py"),
        timeout=7200,
    )

    if not success:
        log("Evaluation FAILED after {} fix attempts".format(fixes))
        update_status("evaluation", "failed")
        return None

    log(stdout[-500:] if stdout else "")

    try:
        with open(output_file, "r") as f:
            results = json.load(f)
        wer_score = results["summary"]["wer"]
        cer_score = results["summary"]["cer"]
        log("WER: {:.1f}% | CER: {:.1f}%".format(wer_score, cer_score))
        update_status("evaluation", "complete", {
            "wer": wer_score,
            "cer": cer_score,
            "model": model_path,
            "fixes_needed": fixes,
        })
        return results["summary"]
    except Exception as e:
        log("Could not parse results: {}".format(e))
        update_status("evaluation", "complete")
        return None


def step_summary(baseline_results, finetuned_results):
    log("=" * 50)
    log("STEP: Summary")

    summary_path = os.path.join(EVAL_DIR, "summary.txt")
    lines = []
    lines.append("=" * 60)
    lines.append("MILO — Training Pipeline Summary")
    lines.append("Date: {}".format(datetime.now().strftime("%Y-%m-%d %H:%M")))
    lines.append("=" * 60)

    if baseline_results:
        lines.append("\nBaseline (whisper-medium, no fine-tuning, 200 samples):")
        lines.append("  WER: {:.1f}%".format(baseline_results["wer"]))
        lines.append("  CER: {:.1f}%".format(baseline_results["cer"]))

    if finetuned_results:
        lines.append("\nFine-tuned model:")
        lines.append("  WER: {:.1f}%".format(finetuned_results["wer"]))
        lines.append("  CER: {:.1f}%".format(finetuned_results["cer"]))
        lines.append("  Model: {}".format(finetuned_results.get("model", "?")))
        lines.append("  Samples: {}".format(finetuned_results.get("num_samples", "?")))

    if baseline_results and finetuned_results:
        improvement = baseline_results["wer"] - finetuned_results["wer"]
        ratio = (improvement / baseline_results["wer"]) * 100 if baseline_results["wer"] > 0 else 0
        lines.append("\nImprovement: {:.1f} pts WER ({:.0f}% relative)".format(improvement, ratio))

    if finetuned_results:
        wer_val = finetuned_results["wer"]
        lines.append("\n--- Recommendation ---")
        if wer_val < 25:
            lines.append("EXCELLENT. Pret pour integration dans le bot.")
            lines.append("Prochaine etape: exporter en faster-whisper ou whisper.cpp")
        elif wer_val < 40:
            lines.append("BON. Utilisable avec post-traitement.")
            lines.append("Ameliorations possibles: augmentation de donnees, plus d'epochs.")
        elif wer_val < 60:
            lines.append("CORRECT. Premiers resultats encourageants.")
            lines.append("A essayer: data augmentation, learning rate scheduling, nettoyage des donnees.")
        else:
            lines.append("INSUFFISANT. Le modele a besoin de plus de travail.")
            lines.append("Pistes: verifier la qualite des donnees, reduire le lr, essayer whisper-small.")

    # Auto-fix history
    if os.path.exists(FIXES_LOG):
        lines.append("\n--- Auto-fixes applied ---")
        with open(FIXES_LOG, "r") as f:
            fix_count = f.read().count("FIX ATTEMPT")
        lines.append("{} fix(es) auto-applique(s). Voir: {}".format(fix_count, FIXES_LOG))

    lines.append("\n--- Fichiers ---")
    lines.append("Resultats detailles: {}/results.json".format(EVAL_DIR))
    lines.append("Log orchestrateur:   {}".format(LOG_FILE))
    lines.append("Historique fixes:    {}".format(FIXES_LOG))
    lines.append("Status pipeline:     {}".format(STATUS_FILE))
    lines.append("\n" + "=" * 60)

    text = "\n".join(lines)
    log("\n" + text)

    os.makedirs(EVAL_DIR, exist_ok=True)
    with open(summary_path, "w") as f:
        f.write(text + "\n")

    update_status("summary", "complete")


# === Main ===

def main():
    parser = argparse.ArgumentParser(description="Milo self-healing pipeline orchestrator")
    parser.add_argument("--skip-training", action="store_true")
    parser.add_argument("--skip-baseline", action="store_true")
    args = parser.parse_args()

    log("=" * 50)
    log("MILO ORCHESTRATOR STARTED (self-healing mode)")
    log("Max auto-fix attempts per step: {}".format(MAX_FIX_ATTEMPTS))
    log("=" * 50)

    update_status("orchestrator", "running")

    # Step 1: Wait for training (with auto-relaunch on error)
    if not args.skip_training:
        if not check_training_running():
            done, result, error_msg = check_training_complete()
            if not done:
                log("ERROR: No training running and no completed training.")
                log("Launch training first: tmux new -s milo '...'")
                update_status("orchestrator", "failed", "no training")
                sys.exit(1)
            elif result == "error":
                log("Previous training failed. Will attempt fix...")
        training_ok = step_wait_training()
        if not training_ok:
            log("Training failed and could not be fixed. Stopping.")
            update_status("orchestrator", "failed", "training unrecoverable")
            sys.exit(1)
    else:
        log("Skipping training (--skip-training)")

    # Step 2: Baseline eval
    baseline_results = None
    if not args.skip_baseline:
        baseline_results = step_evaluate_baseline()

    # Step 3: Fine-tuned eval
    finetuned_results = step_evaluate()

    # Step 4: Summary
    step_summary(baseline_results, finetuned_results)

    update_status("orchestrator", "complete")
    log("=" * 50)
    log("PIPELINE COMPLETE — check {}/summary.txt".format(EVAL_DIR))
    log("=" * 50)


if __name__ == "__main__":
    main()
