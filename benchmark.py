import json
import os
import copy
import argparse
from pathlib import Path
import subprocess
import time
from openai import OpenAI
from urllib.parse import urlsplit, unquote
import problems
from tinydb import TinyDB

import deltagen
from algorithms import BestOfKAlgorithm, BestOfKHyperparams, IterativeAlgorithm, IterativeHyperparams, SimulatedAnneal, SimulatedAnnealHyperparams


DEFAULT_CONTEXT_SIZE = 8192

def get_filename(url: str) -> str:
    path = urlsplit(url).path
    filename = unquote(os.path.basename(path))
    return filename

def download_file(link, output_path):
    os.system(f"wget {link} {output_path}")  # TODO: Safest approach to this problem that ever existed, what could go wrong?

def download_models(data, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    models = data["models"]

    for model in models:
        model_name = model["name"]
        model_link = model["link"]

        filename = get_filename(model_link)
        file_path = os.path.join(output_dir, filename)
        if os.path.exists(file_path):
            deltagen.debug(f"Path {file_path} exists. Skipping download")
            continue

        deltagen.debug(f"Downloading {model_name} to {file_path}")
        success = download_file(model_link, file_path)
        if success:
            deltagen.debug(f"Downloaded {model_name}")
        else:
            deltagen.debug(f"Failed to download {model_name}")

def read_file(path: str) -> str:
    with open(path) as f:
        return f.read()

def load_json_file(file_path):
    return json.loads(read_file(file_path))

A2 = {
    "rate_fn": problems.rate_a2_roi,
    "init_solution": read_file("init_solution.py"),
    "problem": read_file("problem.txt")
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('llama_server_path', type=str)
    args = parser.parse_args()
    json_file_path = "bench_config.json"

    json_data = load_json_file(json_file_path)
    output_dir = json_data.get("model_dir")
    download_models(json_data, output_dir)

    LOAD_MODEL = "load_model"
    BENCH_ALGO = "bench_algo"
    STEP_ALGO = "step_algo"
    INIT_SOMETHING = "init_something"
    EXCEPTION = "exception"

    model_dir = Path(output_dir)

    db = TinyDB(f"data/run_{int(time.time())}.json")
    exec_log = db.table("execution_log")
    results = db.table("bench_results")
    
    for model in json_data["models"]:
        name = model["name"]
        path = Path(model.get('path') or (model_dir / get_filename(model["link"])))
        deltagen.debug(f"Benchmarking model {name}, {path=}")
        deltagen.debug(args.llama_server_path)
        exec_log.insert({"time": time.time(), "action": LOAD_MODEL, "model": model["name"], "start": True})
        llama_server_proc = subprocess.Popen([args.llama_server_path, '-m', path.as_posix(), '-c', str(model.get("ctx_alloc", DEFAULT_CONTEXT_SIZE)), '--port', '8643', "--xtc-probability", "0.5", "--xtc-threshold", "0.1"])
        deltagen.client = OpenAI(api_key="sk-no-key-required", base_url="http://localhost:8643/v1/")
        exec_log.insert({"time": time.time(), "action": LOAD_MODEL, "model": model["name"], "start": False})
        NUM_ITER = 100
        K_INITIAL_BEST_OF_K = 5
        BENCH_PROBLEM = A2
        N_TRIES_FOR_EACH_ALGO = 3
        ALGOS = [
            {"hparams": SimulatedAnnealHyperparams.from_iter_count(int(NUM_ITER*0.85)), "algo_class": SimulatedAnneal, "title": "Метод отжига"},
            {"hparams": IterativeHyperparams(), "algo_class": IterativeAlgorithm, "title": "Итеративный алгоритм (без score_decay)"},
            {"hparams": IterativeHyperparams(score_decay=0.85, decay_cooling_rate=(1.1/0.85)**(1/NUM_ITER)), "algo_class": IterativeAlgorithm, "title": f"Итеративный алгоритм (score_decay=0.85)"},
            {"hparams": BestOfKHyperparams(), "algo_class": BestOfKAlgorithm, "title": "Лучший из K"},
        ] + [
            
#            {"hparams": IterativeHyperparams(score_decay=decay, decay_cooling_rate=(1.1/decay)**(1/NUM_ITER)), "algo_class": IterativeAlgorithm, "title": f"Итеративный алгоритм (score_decay={decay})"} for decay in (1.0, 0.95, 0.9, 0.85, 0.8, 0.7, 0.6, 0.4, 0.2)
            #{"hparams": IterativeHyperparams(score_decay=0.75, decay_cooling_rate=1.03), "algo_class": IterativeAlgorithm, "title": "Итеративный алгоритм (score_decay=0.75)"},
            #{"hparams": IterativeHyperparams(score_decay=0.6, decay_cooling_rate=1.05), "algo_class": IterativeAlgorithm, "title": "Итеративный алгоритм (score_decay=0.6)"},
        ]
        ALGOS2 = ALGOS
        ALGOS = []
        for algo in ALGOS2:
            for i in range(N_TRIES_FOR_EACH_ALGO if algo["algo_class"] is not BestOfKAlgorithm else 1):
                algo_copy = copy.deepcopy(algo)
                algo_copy["title"] = algo_copy["title"] + f" #{i+1}"
                ALGOS.append(algo_copy)
        deltagen.debug("ALGOS TO BENCH:", ALGOS)
            
        problem = BENCH_PROBLEM["problem"]
        init_solution = BENCH_PROBLEM["init_solution"]
        initial_solution = deltagen.CodeSolution(problem=problem, rate_fn=BENCH_PROBLEM["rate_fn"], init_solution=init_solution)
        step_zero_solution = initial_solution
        init_best_of_k = BestOfKAlgorithm(step_zero_solution)
        init_best_of_k_hparams = BestOfKHyperparams()
        for _ in range(K_INITIAL_BEST_OF_K):
            step_zero_solution = init_best_of_k.step(init_best_of_k_hparams, step_zero_solution)

        for algo_entry in copy.deepcopy(ALGOS):
            hparams, algo_class = algo_entry["hparams"], algo_entry["algo_class"]
            title = algo_entry["title"]
            exec_log.insert({"time": time.time(), "action": BENCH_ALGO, "model": model["name"], "algo": title, "start": True})
            exec_log.insert({"time": time.time(), "action": INIT_SOMETHING, "model": model["name"], "algo": title, "start": True, "data": {"init": "solution"}})
            solution = copy.deepcopy(step_zero_solution)
            exec_log.insert({"time": time.time(), "action": INIT_SOMETHING, "model": model["name"], "algo": title, "start": False, "data": {"init": "solution", "solution": str(solution)}})
            exec_log.insert({"time": time.time(), "action": INIT_SOMETHING, "model": model["name"], "algo": title, "start": True, "data": {"init": "algo"}})
            algo = algo_class(solution)
            exec_log.insert({"time": time.time(), "action": INIT_SOMETHING, "model": model["name"], "algo": title, "start": False, "data": {"init": "algo", "algo": str(algo)}})
            try:
                for i in range(NUM_ITER):
                    try:
                        exec_log.insert({"time": time.time(), "action": STEP_ALGO, "model": model["name"], "algo": title, "start": True})
                        solution = algo.step(hparams, solution)
                    except Exception as e:
                        exec_log.insert({"time": time.time(), "action": EXCEPTION, "model": model["name"], "start": True, "data": {EXCEPTION: str(e)}})
                        deltagen.debug(457, e)
                    results.insert({"time": time.time(), "iter": i, "algo": title, "model": model["name"], "temp": (hparams.temperature if algo_class is SimulatedAnneal else -1)} | solution.as_dict())
            finally:
                deltagen.debug(solution)
            exec_log.insert({"time": time.time(), "action": BENCH_ALGO, "model": model["name"], "algo": title, "start": False})

        llama_server_proc.kill()
        time.sleep(5)
        llama_server_proc.terminate()
        llama_server_proc.wait()

if __name__ == "__main__":
    main()
