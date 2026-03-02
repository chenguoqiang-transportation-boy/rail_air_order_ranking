import math
import json
import random
import numpy as np
from copy import deepcopy
from typing import Dict, List, Tuple, Optional

# ✅ 按你的文件名导入
from simulation_pressure_priority_search import Simulator


def log_uniform(low: float, high: float) -> float:
    """在 [low, high] 上按 log-uniform 采样（low>0）"""
    assert low > 0 and high > 0 and high >= low
    return math.exp(random.uniform(math.log(low), math.log(high)))


def sample_weights_logspace() -> Dict[str, float]:
    """
    你目前特征量级差很大，log-space 更稳。
    范围可以再缩小/扩大：先跑起来再调。
    """
    return {
        "regret":   log_uniform(1e-3, 1e3),
        "window":   log_uniform(1e-3, 1e3),
        "scarcity": log_uniform(1e-3, 1e3),
        "tightness":log_uniform(1e-3, 1e3),
        "slope":    log_uniform(1e-3, 1e3),
    }


def run_once(conf: Dict, data_dir: str, file_params: Dict[str, str], weights: Dict[str, float], seed: int) -> Dict:
    random.seed(seed)
    np.random.seed(seed)

    sim = Simulator(
        config=deepcopy(conf),
        data_dir=data_dir,
        file_params=file_params,
        score_weights=weights
    )
    return sim.run()


def objective(results: Dict) -> Tuple[float, Dict]:
    fm = results.get("final_metrics", {})
    total_cost = float(fm.get("total_cost", 1e18))

    info = {
        "total_cost": total_cost,
        "num_unserved": int(fm.get("num_unserved", 0)),
        "num_delayed": int(fm.get("num_delayed", 0)),
        "num_delivered": int(fm.get("num_delivered", 0)),
        "delay_rate": float(fm.get("delay_rate", 0.0)),
        "unserved_rate": float(fm.get("unserved_rate", 0.0)),
    }
    return total_cost, info


def evaluate(conf: Dict, data_dir: str, file_params: Dict[str, str], weights: Dict[str, float], seeds: List[int],
             penalty_unfinished: float = 1e6, penalty_unserved: float = 0.0) -> Tuple[float, Dict]:
    objs = []
    infos = []
    for s in seeds:
        res = run_once(conf, data_dir, file_params, weights, s)
        obj, info = objective(res)
        objs.append(obj)
        infos.append(info)

    avg_obj = sum(objs) / len(objs)
    avg_info = {
        "avg_obj": avg_obj,
        "avg_total_cost": sum(i["total_cost"] for i in infos) / len(infos),
        # "avg_unserved": sum(i["num_unserved"] for i in infos) / len(infos),
        # "avg_unfinished": sum(i["unfinished"] for i in infos) / len(infos),
        # "avg_delay_rate": sum(i["delay_rate"] for i in infos) / len(infos),
        # "avg_unserved_rate": sum(i["unserved_rate"] for i in infos) / len(infos),
    }
    return avg_obj, avg_info


def random_search(conf: Dict, data_dir: str, file_params: Dict[str, str],
                  n_trials: int = 30,
                  seeds: Optional[List[int]] = None,
                  top_k: int = 5,
                  penalty_unfinished: float = 1e6,
                  penalty_unserved: float = 0.0) -> Dict:
    if seeds is None:
        seeds = [0, 1, 2]
    best = None
    top = []

    for i in range(1, n_trials + 1):
        w = sample_weights_logspace()
        avg_obj, avg_info = evaluate(
            conf, data_dir, file_params, w, seeds,
            penalty_unfinished=penalty_unfinished,
            penalty_unserved=penalty_unserved
        )

        rec = {"trial": i, "weights": w, "obj": avg_obj, "info": avg_info}
        top.append(rec)
        top.sort(key=lambda x: x["obj"])
        top = top[:max(top_k, 10)]

        if best is None or avg_obj < best["obj"]:
            best = rec
            print(f"[BEST@{i}] obj={avg_obj:.2f} weights={w} info={avg_info}")
        elif i % 5 == 0:
            print(f"[{i}/{n_trials}] current_best={best['obj']:.2f}")

    return {"best": best, "top": top[:top_k], "seeds": seeds, "n_trials": n_trials}


if __name__ == "__main__":
    # 这里请保持与你当前 main 一致（你文件里就是这些）
    conf = {
        "period_length": 30,
        "planning_horizon": 15 * 60,
        "num_scenarios": 50,
        "costs": {}
    }

    data_dir = "../data/Instance c/"
    file_params = {
        "train": "train_50.csv",
        "train_station": "train_with_station_50.csv",
        "flight": "flight_50.csv",
        "cargo": "cargo_1.csv"
    }

    result = random_search(
        conf=conf,
        data_dir=data_dir,
        file_params=file_params,
        n_trials=100,            # 先跑30组看趋势，再加到100/200
        seeds=[42],        # 3个seed取平均，减小噪声
        top_k=5,
        penalty_unfinished=1e6, # 强制完成（如果你逻辑已保证全完成，这项始终为0）
        penalty_unserved=0.0    # 想更少DUMMY可设置比如 5000 或 20000（按你的成本量级调）
    )

    print("\n=== BEST ===")
    print("obj:", result["best"]["obj"])
    print("weights:", result["best"]["weights"])
    print("info:", result["best"]["info"])

    print("\n=== TOP 5 ===")
    for k, rec in enumerate(result["top"], 1):
        print(k, "obj=", rec["obj"], "weights=", rec["weights"], "info=", rec["info"])

    with open("best_score_weights.json", "w", encoding="utf-8") as f:
        json.dump(result["best"]["weights"], f, ensure_ascii=False, indent=2)

    print("\nSaved best weights to best_score_weights.json")