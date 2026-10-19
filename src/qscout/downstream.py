"""E7: does forecasting help choose qubits? Replay historical origins and score by realized ESP.

For each origin we pick a linear chain of `chain_len` qubits (a stand-in for a circuit layout)
minimizing predicted -log(ESP), where ESP is the product of (1 - readout error) and (1 - sx error)
per qubit and (1 - two-qubit error) per edge. Strategies differ only in which error values they
believe: the last calibration, the model forecast, or the forecast with trust gating (upper
interval bound for elements whose trust tier is "unreliable"). All are scored on the *realized*
next-cycle values. "Oracle" picks with the realized values and bounds what is achievable.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np
import pandas as pd

from .config import Config
from .evaluation.stats import block_bootstrap_ci

ESP_METRICS = ("readout_error", "sx_error", "two_qubit_error")
STRATEGIES = ("last", "forecast", "gated", "oracle")


def _cost(e: np.ndarray) -> np.ndarray:
    return -np.log1p(-np.clip(e, 0.0, 0.999))


def best_chain(
    node_cost: dict[int, float],
    edge_cost: dict[tuple[int, int], float],
    n: int,
    beam: int = 64,
) -> tuple[float, tuple[int, ...]] | None:
    """Beam search for a simple path of n nodes minimizing node + edge cost (edges undirected,
    keyed (a, b) with a < b). Returns (cost, path) or None if no such path exists."""
    adj: dict[int, list[int]] = defaultdict(list)
    for a, b in edge_cost:
        if a in node_cost and b in node_cost:
            adj[a].append(b)
            adj[b].append(a)
    beams = sorted(((c, (q,)) for q, c in node_cost.items()), key=lambda x: (x[0], x[1]))[:beam]
    for _ in range(n - 1):
        cand = []
        for c, path in beams:
            last = path[-1]
            for nb in adj[last]:
                if nb in path:
                    continue
                e = edge_cost[(min(last, nb), max(last, nb))]
                cand.append((c + e + node_cost[nb], path + (nb,)))
        if not cand:
            return None
        cand.sort(key=lambda x: (x[0], x[1]))
        beams = cand[:beam]
    return beams[0]


def _tables(g: pd.DataFrame, col: str) -> tuple[dict[int, float], dict[tuple[int, int], float]]:
    node: dict[int, float] = defaultdict(float)
    seen: dict[int, set] = defaultdict(set)
    edge: dict[tuple[int, int], float] = {}
    for metric, eid, e in zip(g["metric"], g["entity_id"], g[col], strict=True):
        c = float(_cost(np.array([e]))[0])
        if metric == "two_qubit_error":
            a, b = (int(x) for x in eid.split("-"))
            edge[(min(a, b), max(a, b))] = c
        else:
            q = int(eid)
            node[q] += c
            seen[q].add(metric)
    node = {q: c for q, c in node.items() if seen[q] >= {"readout_error", "sx_error"}}
    return node, edge


def _path_cost(path, node, edge) -> float:
    c = sum(node[q] for q in path)
    for a, b in zip(path[:-1], path[1:], strict=True):
        c += edge[(min(a, b), max(a, b))]
    return c


def run_e7(
    preds: pd.DataFrame, cfg: Config, chain_len: int = 5, stride: int = 3, beam: int = 64
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Returns (per-origin table, summary table). `preds` must hold the shortest-horizon
    predictions (columns: backend, group, ts, metric, entity_id, y0, y, p_lgbm, hi_cqr, tier)."""
    p = preds[preds["metric"].isin(ESP_METRICS)].copy()
    if p.empty:
        return pd.DataFrame(), pd.DataFrame()
    p["e_last"] = np.exp(p["y0"])
    p["e_forecast"] = np.exp(p["y0"] + p["p_lgbm"])
    p["e_gated"] = np.where(p["tier"] == "unreliable", np.exp(p["y0"] + p["hi_cqr"]), p["e_forecast"])
    p["e_oracle"] = np.exp(p["y0"] + p["y"])

    rows = []
    for (backend, grp), gb in p.groupby(["backend", "group"]):
        times = np.sort(gb["ts"].unique())[::stride]
        for ts in times:
            g = gb[gb["ts"] == ts]
            real = _tables(g, "e_oracle")
            res = {"backend": backend, "group": grp, "ts": ts}
            ok = True
            for s in STRATEGIES:
                node, edge = real if s == "oracle" else _tables(g, f"e_{s}")
                best = best_chain(node, edge, chain_len, beam)
                if best is None:
                    ok = False
                    break
                try:
                    res[f"cost_{s}"] = _path_cost(best[1], *real)
                except KeyError:
                    ok = False
                    break
            if ok:
                rows.append(res)
    per = pd.DataFrame(rows)
    if per.empty:
        return per, pd.DataFrame()

    out = []
    for grp, d in per.groupby("group"):
        d = d.reset_index(drop=True)
        gap = float((d["cost_last"] - d["cost_oracle"]).mean())
        for name, a, b in (
            ("forecast_vs_last", "last", "forecast"),
            ("gated_vs_last", "last", "gated"),
            ("gated_vs_forecast", "forecast", "gated"),
        ):
            def stat(x: pd.DataFrame, a=a, b=b) -> float:
                return float((x[f"cost_{a}"] - x[f"cost_{b}"]).mean())

            mean, lo, hi = block_bootstrap_ci(d, "ts", stat, cfg.block_len, cfg.n_boot, cfg.seed)
            out.append(
                {
                    "group": grp,
                    "comparison": name,
                    "n_origins": len(d),
                    "mean_gain_neg_log_esp": mean,
                    "ci_lo": lo,
                    "ci_hi": hi,
                    "win_rate": float((d[f"cost_{a}"] > d[f"cost_{b}"]).mean()),
                    "tie_rate": float(np.isclose(d[f"cost_{a}"], d[f"cost_{b}"]).mean()),
                    "share_of_oracle_gap": mean / gap if gap > 0 else float("nan"),
                }
            )
    return per, pd.DataFrame(out)
