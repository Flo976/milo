"""Milo Status API — Expose training pipeline status on local network."""
import json
import os
import re
import subprocess
import threading
import time
from datetime import datetime
from flask import Flask, jsonify, request

app = Flask(__name__)

MILO_DIR = "/home/florent/milo"
STATUS_FILE = os.path.join(MILO_DIR, "pipeline_status.json")
TRAINING_LOG = os.path.join(MILO_DIR, "training.log")
FIXES_LOG = os.path.join(MILO_DIR, "fixes_history.log")
FIX_RESULTS_DIR = os.path.join(MILO_DIR, "fix_results")
MODEL_OUTPUT = os.path.join(MILO_DIR, "models/whisper-mg-v1")
EVAL_DIR = os.path.join(MILO_DIR, "evaluation")

# Track running fix jobs
fix_jobs = {}
fix_lock = threading.Lock()


# === Gathering helpers ===

def get_gpu_info():
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu",
             "--format=csv,noheader,nounits"],
            text=True, timeout=5,
        ).strip()
        parts = [p.strip() for p in out.split(",")]
        return {
            "gpu_util_pct": int(parts[0]),
            "vram_used_mb": int(parts[1]),
            "vram_total_mb": int(parts[2]),
            "temp_c": int(parts[3]),
        }
    except Exception:
        return None


def get_training_progress():
    """Parse tmux pane output for live training progress."""
    try:
        out = subprocess.check_output(
            ["tmux", "capture-pane", "-t", "milo", "-p", "-S", "-20"],
            text=True, timeout=5,
        ).strip()
    except Exception:
        out = ""

    info = {"raw_last_lines": out.split("\n")[-5:]}

    tqdm_match = re.findall(
        r"(\d+)%\|.*?\|\s*(\d+)/(\d+)\s*\[([^\]]+)<([^\],]+),\s*([^\]]+)\]",
        out,
    )
    if tqdm_match:
        last = tqdm_match[-1]
        info["pct"] = int(last[0])
        info["step"] = int(last[1])
        info["total_steps"] = int(last[2])
        info["elapsed"] = last[3]
        info["eta"] = last[4]
        info["speed"] = last[5]

    loss_matches = re.findall(r"'loss':\s*([\d.]+)", out)
    step_matches = re.findall(r"'learning_rate':\s*([\d.e-]+)", out)
    if loss_matches:
        info["last_loss"] = float(loss_matches[-1])
    if step_matches:
        info["learning_rate"] = float(step_matches[-1])

    if "TRAINING COMPLETE" in out:
        info["status"] = "complete"
    elif "Map:" in out and "Starting training" not in out:
        map_pct = re.findall(r"Map:.*?(\d+)%", out)
        if map_pct:
            info["status"] = "preprocessing"
            info["preprocess_pct"] = int(map_pct[-1])
    elif tqdm_match:
        info["status"] = "training"
    else:
        info["status"] = "unknown"

    return info


def get_orchestrator_status():
    try:
        out = subprocess.check_output(
            ["tmux", "capture-pane", "-t", "orchestrator", "-p", "-S", "-10"],
            text=True, timeout=5,
        ).strip()
        return {"running": True, "last_lines": out.split("\n")[-5:]}
    except Exception:
        return {"running": False}


def get_pipeline_status():
    try:
        with open(STATUS_FILE, "r") as f:
            return json.load(f)
    except Exception:
        return None


def get_checkpoints():
    try:
        dirs = sorted([
            d for d in os.listdir(MODEL_OUTPUT)
            if d.startswith("checkpoint-")
        ], key=lambda x: int(x.split("-")[1]) if x.split("-")[1].isdigit() else 0)
        return dirs
    except Exception:
        return []


def get_eval_results():
    results = {}
    for name in ["baseline_results.json", "results.json"]:
        path = os.path.join(EVAL_DIR, name)
        if os.path.exists(path):
            try:
                with open(path, "r") as f:
                    data = json.load(f)
                results[name.replace(".json", "")] = data.get("summary", data)
            except Exception:
                pass
    return results


def count_fixes():
    try:
        with open(FIXES_LOG, "r") as f:
            return f.read().count("FIX ATTEMPT")
    except Exception:
        return 0


# === Claude fix engine ===

def gather_context():
    """Collect full system context for Claude."""
    ctx = []

    # GPU
    gpu = get_gpu_info()
    if gpu:
        ctx.append("GPU: {}% util, {} / {} MB VRAM, {}°C".format(
            gpu["gpu_util_pct"], gpu["vram_used_mb"], gpu["vram_total_mb"], gpu["temp_c"]))

    # Training progress
    training = get_training_progress()
    ctx.append("Training status: {}".format(training.get("status", "unknown")))
    if training.get("step"):
        ctx.append("Step: {}/{} ({}%)".format(
            training["step"], training.get("total_steps"), training.get("pct")))
    if training.get("last_loss"):
        ctx.append("Last loss: {}".format(training["last_loss"]))

    # Tmux panes
    for session in ["milo", "orchestrator"]:
        try:
            out = subprocess.check_output(
                ["tmux", "capture-pane", "-t", session, "-p", "-S", "-50"],
                text=True, timeout=5,
            ).strip()
            ctx.append("\n--- tmux '{}' (last 50 lines) ---\n{}".format(session, out))
        except Exception:
            ctx.append("\ntmux '{}': not running".format(session))

    # Training log tail
    if os.path.exists(TRAINING_LOG):
        try:
            with open(TRAINING_LOG, "r") as f:
                lines = f.readlines()
            ctx.append("\n--- training.log (last 50 lines) ---\n{}".format(
                "".join(lines[-50:])))
        except Exception:
            pass

    # Orchestrator log tail
    orch_log = os.path.join(MILO_DIR, "orchestrator.log")
    if os.path.exists(orch_log):
        try:
            with open(orch_log, "r") as f:
                lines = f.readlines()
            ctx.append("\n--- orchestrator.log (last 30 lines) ---\n{}".format(
                "".join(lines[-30:])))
        except Exception:
            pass

    # Checkpoints
    checkpoints = get_checkpoints()
    ctx.append("\nCheckpoints: {}".format(checkpoints if checkpoints else "none"))

    # Running processes
    try:
        out = subprocess.check_output(
            ["bash", "-c", "ps aux | grep -E 'python|train|eval' | grep -v grep"],
            text=True, timeout=5,
        ).strip()
        ctx.append("\n--- Relevant processes ---\n{}".format(out))
    except Exception:
        pass

    return "\n".join(ctx)


def run_claude_fix(job_id, problem, mode):
    """Run Claude headless in a thread."""
    os.makedirs(FIX_RESULTS_DIR, exist_ok=True)
    result_file = os.path.join(FIX_RESULTS_DIR, "{}.json".format(job_id))

    with fix_lock:
        fix_jobs[job_id]["status"] = "running"
        fix_jobs[job_id]["started"] = datetime.now().isoformat()

    context = gather_context()

    if mode == "check":
        instruction = """Analyse la situation et reponds avec un diagnostic.
NE MODIFIE AUCUN FICHIER. Fais uniquement un Read des fichiers pertinents.
Reponds avec:
1. ETAT: est-ce que tout va bien ou pas ?
2. PROBLEME: si quelque chose ne va pas, quoi exactement ?
3. RECOMMENDATION: que faut-il faire ?"""
    else:
        instruction = """Diagnostique le probleme et corrige-le directement.
1. Lis les fichiers pertinents
2. Identifie la cause
3. Corrige les fichiers avec Edit/Write
4. Si un process doit etre relance, fais-le via Bash
5. Reponds avec un resume de ce que tu as fait"""

    tools = "Read,Grep,Glob" if mode == "check" else "Edit,Read,Bash,Write,Grep,Glob"

    prompt = """Tu es le systeme de maintenance du projet Milo (fine-tuning Whisper pour le malagasy).

=== PROBLEME SIGNALE ===
{problem}

=== CONTEXTE SYSTEME ===
{context}

=== PROJET ===
- Dossier: /home/florent/milo
- Scripts: /home/florent/milo/scripts/
- Venv: /home/florent/milo/venv
- GPU: RTX 5070 Ti, CUDA 13.1, PyTorch nightly cu128
- Dataset: badrex/malagasy-speech-full (28371 train, 3099 val, 3101 test)
- Training: whisper-medium, 10 epochs, batch 4x4, lr 1e-5
- datasets==2.21 (pas torchcodec)
- tmux sessions: milo (training), orchestrator (monitoring), api (status HTTP)

=== INSTRUCTIONS ===
{instruction}""".format(
        problem=problem,
        context=context,
        instruction=instruction,
    )

    prompt_file = "/tmp/milo_api_fix_{}.txt".format(job_id)
    with open(prompt_file, "w") as f:
        f.write(prompt)

    cmd = 'cat {} | /usr/bin/claude -p --allowedTools "{}" 2>&1'.format(prompt_file, tools)

    try:
        result = subprocess.run(
            ["bash", "-c", cmd],
            capture_output=True, text=True, timeout=600,
        )
        response = result.stdout + result.stderr
        success = result.returncode == 0
    except subprocess.TimeoutExpired:
        response = "TIMEOUT: Claude did not respond within 10 minutes"
        success = False
    except Exception as e:
        response = "ERROR: {}".format(str(e))
        success = False

    # Save result
    job_result = {
        "job_id": job_id,
        "mode": mode,
        "problem": problem,
        "success": success,
        "response": response,
        "timestamp": datetime.now().isoformat(),
    }

    with open(result_file, "w") as f:
        json.dump(job_result, f, indent=2, ensure_ascii=False)

    # Log to fixes history
    with open(FIXES_LOG, "a") as f:
        f.write("\n" + "=" * 70 + "\n")
        f.write("API FIX — {} — mode={}\n".format(datetime.now().isoformat(), mode))
        f.write("Problem: {}\n".format(problem))
        f.write("Success: {}\n".format(success))
        f.write("Response:\n{}\n".format(response[:3000]))
        f.write("=" * 70 + "\n")

    with fix_lock:
        fix_jobs[job_id]["status"] = "done" if success else "failed"
        fix_jobs[job_id]["finished"] = datetime.now().isoformat()
        fix_jobs[job_id]["result_file"] = result_file

    # Cleanup prompt file
    try:
        os.remove(prompt_file)
    except Exception:
        pass


# === Routes ===

@app.route("/")
def index():
    return jsonify({
        "service": "milo-status",
        "endpoints": ["/status", "/gpu", "/training", "/eval", "/fix", "/fix/<id>", "/fix/history", "/doc"],
    })


@app.route("/doc")
def doc():
    return jsonify({
        "service": "Milo Status API",
        "description": "API de monitoring et maintenance du pipeline Milo (fine-tuning Whisper pour le malagasy)",
        "endpoints": {
            "GET /": {
                "description": "Liste des endpoints disponibles",
            },
            "GET /doc": {
                "description": "Cette documentation",
            },
            "GET /status": {
                "description": "Status complet du pipeline en un seul appel",
                "response": {
                    "phase": "Phase actuelle: preprocessing | training | training_done | evaluating | evaluation_done | unknown",
                    "training.status": "Status du training: preprocessing | training | complete | unknown",
                    "training.step": "Step actuel (int ou null)",
                    "training.total_steps": "Total de steps (17740 pour 10 epochs)",
                    "training.pct": "Progression en %",
                    "training.eta": "Temps restant estime (ex: '9:14:11')",
                    "training.speed": "Vitesse (ex: '2.13s/it')",
                    "training.last_loss": "Derniere loss loggee (float, doit diminuer)",
                    "gpu": "Objet GPU (voir GET /gpu)",
                    "checkpoints": "Liste des checkpoints sauvegardes (ex: ['checkpoint-1000', 'checkpoint-2000'])",
                    "eval": "Resultats d'evaluation (voir GET /eval)",
                    "orchestrator": "true si l'orchestrateur tourne",
                    "auto_fixes": "Nombre de corrections auto appliquees",
                },
            },
            "GET /gpu": {
                "description": "Utilisation GPU en temps reel",
                "response": {
                    "gpu_util_pct": "Utilisation GPU en % (normal: 85-100 pendant training)",
                    "vram_used_mb": "VRAM utilisee en MB",
                    "vram_total_mb": "VRAM totale en MB (16303 pour RTX 5070 Ti)",
                    "temp_c": "Temperature en °C (alerte si > 85)",
                },
            },
            "GET /training": {
                "description": "Progres detaille du training, parse depuis le tmux live",
                "response": {
                    "status": "preprocessing | training | complete | unknown",
                    "step": "Step actuel",
                    "total_steps": "Total steps",
                    "pct": "Pourcentage",
                    "elapsed": "Temps ecoule (ex: '1:18:33')",
                    "eta": "Temps restant (ex: '9:14:11')",
                    "speed": "Vitesse par step (ex: '2.13s/it')",
                    "last_loss": "Derniere loss",
                    "learning_rate": "Learning rate actuel",
                    "raw_last_lines": "5 dernieres lignes brutes du terminal",
                },
            },
            "GET /eval": {
                "description": "Resultats d'evaluation (disponibles apres le training)",
                "response": {
                    "baseline_results": "WER/CER du modele whisper-medium sans fine-tuning (null si pas encore evalue)",
                    "results": "WER/CER du modele fine-tune (null si pas encore evalue)",
                },
                "notes": "Chaque resultat contient: wer (%), cer (%), num_samples, eval_time_s, model",
            },
            "POST /fix": {
                "description": "Declenche un Claude Code headless pour diagnostiquer ou corriger un probleme",
                "request_body": {
                    "problem": "(requis) Description du probleme en texte libre",
                    "mode": "(optionnel) 'check' = diagnostic read-only (defaut), 'fix' = diagnostic + correction automatique",
                },
                "response": {
                    "job_id": "ID du job (ex: '20260215_091234')",
                    "status": "queued",
                    "check_result": "URL pour recuperer le resultat: /fix/<job_id>",
                },
                "http_codes": {
                    "202": "Job accepte et lance en arriere-plan",
                    "400": "Champ 'problem' manquant ou mode invalide",
                    "409": "Un fix est deja en cours (un seul a la fois)",
                },
                "examples": [
                    "curl -X POST :5555/fix -H 'Content-Type: application/json' -d '{\"problem\": \"le training semble bloque\"}'",
                    "curl -X POST :5555/fix -H 'Content-Type: application/json' -d '{\"problem\": \"le training a crashe\", \"mode\": \"fix\"}'",
                ],
                "notes": "Le mode 'check' donne a Claude uniquement Read/Grep/Glob. Le mode 'fix' ajoute Edit/Write/Bash. Timeout: 10 min. Contexte systeme (GPU, logs, tmux, processes) collecte automatiquement.",
            },
            "GET /fix/<job_id>": {
                "description": "Recupere le resultat d'un job de fix",
                "response": {
                    "job_id": "ID du job",
                    "problem": "Description du probleme soumis",
                    "mode": "check ou fix",
                    "status": "queued | running | done | failed",
                    "result.success": "true/false",
                    "result.response": "Reponse complete de Claude",
                },
                "http_codes": {
                    "200": "Job trouve",
                    "404": "Job introuvable",
                },
            },
            "GET /fix/history": {
                "description": "Liste tous les jobs de fix (en cours et passes)",
                "response": "Array de jobs avec job_id, problem, mode, status, timestamps",
            },
        },
        "project_context": {
            "model": "whisper-medium (769M params) fine-tune sur donnees malagasy",
            "dataset": "badrex/malagasy-speech-full — 28371 train / 3099 val / 3101 test (~166h audio)",
            "gpu": "RTX 5070 Ti, 16 Go VRAM",
            "target_wer": "< 40% (langue low-resource, baseline 106%)",
            "tmux_sessions": {
                "milo": "Training en cours",
                "orchestrator": "Monitoring + auto-eval apres training",
                "api": "Cette API",
            },
        },
    })


@app.route("/status")
def status():
    training = get_training_progress()
    gpu = get_gpu_info()
    pipeline = get_pipeline_status()
    orchestrator = get_orchestrator_status()
    checkpoints = get_checkpoints()
    evals = get_eval_results()
    fixes = count_fixes()

    phase = "unknown"
    if training.get("status") == "complete":
        phase = "training_done"
    elif training.get("status") == "training":
        phase = "training"
    elif training.get("status") == "preprocessing":
        phase = "preprocessing"

    if evals.get("results"):
        phase = "evaluation_done"
    elif pipeline and pipeline.get("steps", {}).get("evaluation", {}).get("status") == "in_progress":
        phase = "evaluating"

    summary = {
        "phase": phase,
        "training": {
            "status": training.get("status"),
            "step": training.get("step"),
            "total_steps": training.get("total_steps"),
            "pct": training.get("pct"),
            "eta": training.get("eta"),
            "speed": training.get("speed"),
            "last_loss": training.get("last_loss"),
        },
        "gpu": gpu,
        "checkpoints": checkpoints,
        "eval": evals,
        "orchestrator": orchestrator.get("running"),
        "auto_fixes": fixes,
    }
    return jsonify(summary)


@app.route("/gpu")
def gpu():
    return jsonify(get_gpu_info())


@app.route("/training")
def training():
    return jsonify(get_training_progress())


@app.route("/eval")
def evaluation():
    return jsonify(get_eval_results())


@app.route("/fix", methods=["POST"])
def trigger_fix():
    """Trigger a Claude check or fix.

    POST JSON body:
        problem: str  — description of the issue (required)
        mode: str     — "check" (read-only diagnosis) or "fix" (auto-correct)
                        default: "check"

    Examples:
        curl -X POST http://host:5555/fix -H 'Content-Type: application/json' \
             -d '{"problem": "le training semble bloque"}'

        curl -X POST http://host:5555/fix -H 'Content-Type: application/json' \
             -d '{"problem": "le training a crashe", "mode": "fix"}'
    """
    data = request.get_json(force=True, silent=True) or {}
    problem = data.get("problem", "").strip()
    mode = data.get("mode", "check").strip().lower()

    if not problem:
        return jsonify({"error": "missing 'problem' field"}), 400

    if mode not in ("check", "fix"):
        return jsonify({"error": "mode must be 'check' or 'fix'"}), 400

    # Check no other fix is currently running
    with fix_lock:
        running = [j for j in fix_jobs.values() if j["status"] == "running"]
        if running:
            return jsonify({
                "error": "a fix is already running",
                "running_job": running[0],
            }), 409

    job_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    with fix_lock:
        fix_jobs[job_id] = {
            "job_id": job_id,
            "problem": problem,
            "mode": mode,
            "status": "queued",
            "queued": datetime.now().isoformat(),
        }

    thread = threading.Thread(target=run_claude_fix, args=(job_id, problem, mode))
    thread.daemon = True
    thread.start()

    return jsonify({
        "job_id": job_id,
        "mode": mode,
        "status": "queued",
        "check_result": "/fix/{}".format(job_id),
    }), 202


@app.route("/fix/<job_id>")
def get_fix_result(job_id):
    """Get the result of a fix job."""
    with fix_lock:
        job = fix_jobs.get(job_id)

    if not job:
        return jsonify({"error": "job not found"}), 404

    result = dict(job)

    # If done, load full response from file
    if job.get("result_file") and os.path.exists(job["result_file"]):
        try:
            with open(job["result_file"], "r") as f:
                result["result"] = json.load(f)
        except Exception:
            pass

    return jsonify(result)


@app.route("/fix/history")
def fix_history():
    """List all fix jobs."""
    with fix_lock:
        return jsonify(list(fix_jobs.values()))


if __name__ == "__main__":
    os.makedirs(FIX_RESULTS_DIR, exist_ok=True)
    print("=" * 50)
    print("MILO STATUS API")
    print("=" * 50)
    print("Endpoints:")
    print("  GET  /status       — Full pipeline status")
    print("  GET  /gpu          — GPU utilization")
    print("  GET  /training     — Training progress")
    print("  GET  /eval         — Evaluation results")
    print("  POST /fix          — Trigger Claude check/fix")
    print("  GET  /fix/<id>     — Get fix result")
    print("  GET  /fix/history  — All fix jobs")
    print()
    print("Fix usage:")
    print('  curl -X POST :5555/fix -H "Content-Type: application/json" \\')
    print('       -d \'{"problem": "description", "mode": "check|fix"}\'')
    print("=" * 50)
    app.run(host="0.0.0.0", port=5555, debug=False)
