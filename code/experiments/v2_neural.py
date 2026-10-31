"""Train/validation-only nonlinear comparisons. Never load the test split.

The exported NumPy scorer accepts arbitrary bank sizes and uses no references.
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
import argparse
import json
from pathlib import Path
import sys
import time
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import DATA, load_inputs, read_jsonl
from student_features import prepare_inputs, group_correctness, choose_from_scores
from student_model import fit_ridge, score_candidates


def load(split):
    if split not in ("train", "validation"):
        raise ValueError("Only training and validation are permitted")
    bank, arrays = load_inputs(split)
    answers, groups, emb, text = prepare_inputs(bank, arrays)
    refs = read_jsonl(DATA / f"gsm8k_{split}.jsonl")
    assert [r["id"] for r in refs] == [r["id"] for r in bank]
    return groups, np.concatenate([emb, text], axis=-1), group_correctness(answers, refs)


class Verifier(nn.Module):
    def __init__(self, d, hidden, dropout):
        super().__init__()
        self.fc1 = nn.Linear(d, hidden)
        self.fc2 = nn.Linear(hidden, hidden // 2)
        self.out = nn.Linear(hidden // 2, 1)
        self.drop = nn.Dropout(dropout)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, x, base):
        h = self.drop(F.gelu(self.fc1(x), approximate="tanh"))
        h = self.drop(F.gelu(self.fc2(h), approximate="tanh"))
        return base + self.out(h).squeeze(-1)


def preprocess(fit, query, config, stats=None):
    if stats is None:
        raw_mean = fit.mean(axis=(0, 1))
        raw_scale = np.maximum(fit.std(axis=(0, 1)), .05)
        diff = fit - fit.mean(axis=1, keepdims=True)
        diff_scale = np.maximum(diff.std(axis=(0, 1)), .05)
        stats = dict(raw_mean=raw_mean, raw_scale=raw_scale, diff_scale=diff_scale)
    raw = np.clip((query - stats["raw_mean"]) / stats["raw_scale"], -8, 8)
    diff = np.clip((query - query.mean(axis=1, keepdims=True)) / stats["diff_scale"], -8, 8)
    if config["mode"] == "raw":
        x = raw
    elif config["mode"] == "diff":
        x = diff
    else:
        x = np.concatenate([raw, diff], axis=-1)
    return x.astype(np.float32), stats


def correct_count(scores, groups, truth, vote):
    choices = choose_from_scores(groups, scores, vote)
    hard = np.array([len(g) >= 2 for g in groups])
    idx = np.where(hard)[0]
    return int(truth[idx, np.asarray(choices)[hard]].sum())


def train(config, features, truth, groups, val_f, val_y, val_g, epochs=None):
    torch.manual_seed(config["seed"])
    weights, scale, _ = fit_ridge(features, truth, groups, .3)
    x, stats = preprocess(features, features, config)
    vx, _ = preprocess(features, val_f, config, stats)
    tb = config["base_scale"] * score_candidates(features, weights, scale)
    vb = config["base_scale"] * score_candidates(val_f, weights, scale)
    device = config.get("device", "cpu")
    model = Verifier(x.shape[-1], config["hidden"], config["dropout"]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=config["lr"], weight_decay=config["wd"])
    tx = torch.from_numpy(x).to(device)
    ty = torch.from_numpy(truth.astype(np.float32)).to(device)
    tbase = torch.from_numpy(tb.astype(np.float32)).to(device)
    vtx = torch.from_numpy(vx).to(device)
    vtbase = torch.from_numpy(vb.astype(np.float32)).to(device)
    best = None
    history = []
    started = time.perf_counter()
    maxepoch = epochs or config["epochs"]
    for epoch in range(1, maxepoch + 1):
        model.train()
        losses = []
        for idx in torch.randperm(len(tx), device=device).split(config["batch_size"]):
            scores = model(tx[idx], tbase[idx])
            labels = ty[idx]
            if config["loss"] == "mass":
                loss = (torch.logsumexp(scores, -1) - torch.logsumexp(scores.masked_fill(labels < .5, -1e9), -1)).mean()
            else:
                target = labels / labels.sum(-1, keepdim=True)
                loss = -(target * F.log_softmax(scores, -1)).sum(-1).mean()
            loss = loss + config["bce"] * F.binary_cross_entropy_with_logits(scores, labels)
            loss = loss + config["penalty"] * (scores - tbase[idx]).square().mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
            opt.step()
            losses.append(float(loss.detach()))
        if epoch % 5 == 0 or epoch == maxepoch:
            model.eval()
            with torch.no_grad():
                pred = model(vtx, vtbase).cpu().numpy()
            candidates = [(correct_count(pred, val_g, val_y, v), v) for v in [0., .2, .4, .6]]
            count, vote = max(candidates, key=lambda row: row[0])
            history.append(dict(epoch=epoch, hard_correct=count, vote=vote, loss=float(np.mean(losses))))
            if epochs is not None or best is None or count > best["correct"]:
                params = {k: p.detach().cpu().numpy().copy() for k, p in model.state_dict().items()}
                best = dict(correct=count, vote=vote, epoch=epoch, scores=pred.copy(), params=params)
    best.update(stats=stats, ridge_weights=weights, ridge_scale=scale,
                seconds=time.perf_counter() - started, history=history, config=config)
    return best


def export(best, path):
    path.mkdir(parents=True, exist_ok=True)
    np.savez(path / "weights.npz", **best["params"], **best["stats"], ridge_weights=best["ridge_weights"], ridge_scale=best["ridge_scale"])
    (path / "config.json").write_text(json.dumps({k: best[k] for k in ["config", "epoch", "vote", "correct"]}, indent=2), encoding="utf-8")


def numpy_scores(features, checkpoint):
    meta = json.loads((Path(checkpoint) / "config.json").read_text())
    p = np.load(Path(checkpoint) / "weights.npz")
    config = meta["config"]
    x, _ = preprocess(features, features, config, p)
    base = config["base_scale"] * score_candidates(features, p["ridge_weights"], p["ridge_scale"])
    def gelu(a):
        return .5 * a * (1 + np.tanh(np.sqrt(2 / np.pi) * (a + .044715 * a ** 3)))
    h = gelu(x @ p["fc1.weight"].T + p["fc1.bias"])
    h = gelu(h @ p["fc2.weight"].T + p["fc2.bias"])
    return base + (h @ p["out.weight"].T + p["out.bias"]).squeeze(-1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cv", action="store_true")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--retrain-from", type=Path,
                        help="Replay frozen config and epoch without selecting a new model")
    parser.add_argument("--output", type=Path, default=Path("runs/v2_neural"))
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    started = time.perf_counter()
    g, f, y = load("train")
    vg, vf, vy = load("validation")
    args.output.mkdir(parents=True, exist_ok=True)
    if args.retrain_from is not None:
        metadata = json.loads(args.retrain_from.read_text(encoding="utf-8"))
        config = metadata["config"]
        result = train(config, f, y, g, vf, vy, vg, epochs=metadata["epoch"])
        result["vote"] = metadata["vote"]
        export(result, args.output / "bundle")
        np.savez(args.output / "validation_scores.npz", scores=result["scores"])
        summary = {k: result[k] for k in ["config", "epoch", "vote", "correct", "seconds", "history"]}
        (args.output / "retrain_record.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(summary, flush=True)
        return
    configs = []
    base = dict(hidden=128, dropout=.35, lr=.0005, wd=.15, batch_size=128,
                epochs=60, seed=20261030, base_scale=4., penalty=.02, bce=.2,
                mode="both", loss="soft", device=args.device)
    for override in [dict(), dict(mode="raw", bce=.4), dict(mode="diff", bce=0.), dict(hidden=256, loss="mass", bce=.2)]:
        configs.append({**base, **override})
    records = []
    best = None
    if not args.cv:
        for i, config in enumerate(configs):
            result = train(config, f, y, g, vf, vy, vg)
            print(i, result["correct"], result["vote"], result["epoch"], round(result["seconds"], 2), flush=True)
            record = {k: result[k] for k in ["config", "epoch", "vote", "correct", "seconds", "history"]}
            records.append(record)
            if best is None or result["correct"] > best["correct"]:
                best = result
        export(best, args.output / "bundle")
        np.savez(args.output / "validation_scores.npz", scores=best["scores"])
        replay = numpy_scores(vf, args.output / "bundle")
        print("max_replay_error", np.max(np.abs(replay - best["scores"])), flush=True)
        summary = dict(trials=records, total_seconds=time.perf_counter() - started, torch=torch.__version__, numpy=np.__version__, threads=4,
                       best={k: best[k] for k in ["config", "epoch", "vote", "correct"]})
        (args.output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    else:
        meta = json.loads((args.output / "bundle" / "config.json").read_text())
        config = meta["config"]
        config["device"] = args.device
        hard = np.flatnonzero([len(row) >= 2 for row in g])
        folds = np.array_split(np.random.default_rng(20261030).permutation(hard), 5)
        scores = np.full((len(f), 4), np.nan, np.float32)
        cvrecords = []
        for fold, valid in enumerate(folds):
            fit = np.setdiff1d(np.arange(len(f)), valid)
            r = train(config, f[fit], y[fit], [g[i] for i in fit], f[valid], y[valid], [g[i] for i in valid], epochs=meta["epoch"])
            scores[valid] = r["scores"]
            count = correct_count(r["scores"], [g[i] for i in valid], y[valid], meta["vote"])
            print("fold", fold, count, len(valid), flush=True)
            cvrecords.append(dict(fold=fold, n=len(valid), correct=count, seconds=r["seconds"]))
        np.savez(args.output / "oof_scores.npz", scores=scores, indices=hard)
        summary = dict(folds=cvrecords, correct=sum(x["correct"] for x in cvrecords), n=len(hard), seed=20261030, seconds=time.perf_counter() - started, config=config, epoch=meta["epoch"], vote=meta["vote"])
        (args.output / "cv_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(summary, flush=True)


if __name__ == "__main__":
    main()
