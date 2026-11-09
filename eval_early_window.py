#!/usr/bin/env python3
"""
eval_early_window.py — EXP-E: Early Observation Window Sensitivity (L=20, L=50 vs L=100)

Evaluates TinyBatteryNet performance when truncated to early cycle observation windows
L ∈ {20, 50, 100} cycles to assess early RUL estimation capability.

Responds to: Reviewer feedback on observation window sensitivity / early prediction capability.
Output: tables/table_early_window_sensitivity.tex
        results_log.json
"""
import os, sys, json, types, importlib
from pathlib import Path
import joblib, numpy as np, torch

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
os.chdir(str(_HERE))

from data_provider.data_factory import data_provider_baseline
from main import compute_metrics, DOMAINS, PAPER_BUNDLE_DIR, CHECKPOINTS

PAPER_CKPT = {
    "Li-ion": "TinyBatteryNetV1R_sl1_lr0.0003_dm64_nh4_el2_dl1_df16_lradjplateau_datasetMIX_large_lossMAPE_wd0.0001_wlTrue_bs32_s42-TinyBatteryNetV1R",
    "Zn-ion": "TinyBatteryNetV1R_sl1_lr0.0001_dm64_nh4_el2_dl1_df16_lradjplateau_datasetZN-coin_lossMAPE_wd0.003_wlTrue_bs32_s42-TinyBatteryNetV1R",
    "Na-ion": "TinyBatteryNetV1R_sl1_lr0.0003_dm64_nh4_el2_dl1_df16_lradjplateau_datasetNA-ion_lossMAPE_wd0.0001_wlTrue_bs16_s42-TinyBatteryNetV1R_na2092_retrain",
    "CALB":   "TinyBatteryNetV1R_sl1_lr0.00045_dm64_nh4_el2_dl1_df16_lradjconstant_datasetCALB_lossMSE_wd0.0005_wlFalse_bs16_s42-TinyBatteryNetV1R",
}

def load_model(ckpt_dir):
    with open(ckpt_dir / "args.json") as f:
        raw = json.load(f)
    args = types.SimpleNamespace(**raw)
    args.alpha1 = getattr(args, "alpha1", 0.15)
    args.alpha2 = getattr(args, "alpha2", 0.10)
    model = importlib.import_module(f"models.{args.model}").Model(args).float()
    state = torch.load(ckpt_dir / "checkpoint_last.pt", map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.eval()
    ls = joblib.load(ckpt_dir / "label_scaler") if (ckpt_dir / "label_scaler").exists() else None
    lcs = joblib.load(ckpt_dir / "life_class_scaler") if (ckpt_dir / "life_class_scaler").exists() else None
    return model, args, ls, lcs

def eval_truncated(model, args, ls, lcs, max_cycles):
    test_data, test_loader = data_provider_baseline(args, "test", None, ls, life_class_scaler=lcs)
    std_val = float(np.sqrt(test_data.label_scaler.var_[-1]))
    mean_val = float(test_data.label_scaler.mean_[-1])
    preds, refs, seen_unseen = [], [], []
    with torch.no_grad():
        for batch in test_loader:
            ccd, cam, lbl, _lc, _slc, _w, su_ids = batch
            ccd = ccd.float()
            cam = cam.float()
            # Truncate mask beyond max_cycles
            if max_cycles < 100:
                cam[:, max_cycles:] = 0.0

            out = model(ccd, cam)
            p = (out * std_val + mean_val).cpu().numpy().reshape(-1)
            r = (lbl * std_val + mean_val).numpy().reshape(-1)
            preds.extend(p.tolist())
            refs.extend(r.tolist())
            seen_unseen.extend(su_ids.numpy().reshape(-1).tolist())
    return compute_metrics(preds, refs, seen_unseen)

def main():
    print("=" * 60)
    print("EXP-E: Early Observation Window Sensitivity (L=20, L=50, L=100)")
    print("=" * 60)

    results = {}
    windows = [20, 50, 100]

    for domain in DOMAINS:
        name = PAPER_CKPT[domain]
        ckpt_dir = Path(PAPER_BUNDLE_DIR) / name
        if not ckpt_dir.exists():
            ckpt_dir = Path(CHECKPOINTS) / name
        if not ckpt_dir.exists():
            print(f"Skipping {domain}: checkpoint not found")
            continue

        print(f"\nEvaluating {domain}...")
        model, args, ls, lcs = load_model(ckpt_dir)
        results[domain] = {}

        for L in windows:
            m = eval_truncated(model, args, ls, lcs, max_cycles=L)
            results[domain][f"L_{L}"] = {
                "mape": m["mape"],
                "acc1": m["acc1"],
                "mae": m["mae"],
            }
            print(f"  L = {L:3d} cycles | MAPE = {m['mape']*100:.2f}% | Acc@15% = {m['acc1']*100:.2f}% | MAE = {m['mae']:.2f}")

    os.makedirs("tables", exist_ok=True)
    tex_path = "tables/table_early_window_sensitivity.tex"
    with open(tex_path, "w") as f:
        f.write("% Early Observation Window Sensitivity Table\n")
        f.write("\\begin{table}[htbp]\n\\centering\n")
        f.write("\\caption{Prediction accuracy sensitivity to early cycle observation window length $L \\in \\{20, 50, 100\\}$ cycles.}\n")
        f.write("\\label{tab:early_window_sensitivity}\n")
        f.write("\\begin{tabular}{l ccc ccc ccc}\n\\toprule\n")
        f.write("Domain & \\multicolumn{3}{c}{$L=20$ Cycles} & \\multicolumn{3}{c}{$L=50$ Cycles} & \\multicolumn{3}{c}{$L=100$ Cycles (Default)} \\\\\n")
        f.write("\\cmidrule(lr){2-4} \\cmidrule(lr){5-7} \\cmidrule(lr){8-10}\n")
        f.write(" & MAPE (\\%) & Acc@15\\% & MAE & MAPE (\\%) & Acc@15\\% & MAE & MAPE (\\%) & Acc@15\\% & MAE \\\\\n\\midrule\n")
        for dom in DOMAINS:
            if dom in results:
                r20 = results[dom]["L_20"]
                r50 = results[dom]["L_50"]
                r100 = results[dom]["L_100"]
                f.write(f"{dom} & {r20['mape']*100:.2f} & {r20['acc1']*100:.1f}\\% & {r20['mae']:.1f} & "
                        f"{r50['mape']*100:.2f} & {r50['acc1']*100:.1f}\\% & {r50['mae']:.1f} & "
                        f"{r100['mape']*100:.2f} & {r100['acc1']*100:.1f}\\% & {r100['mae']:.1f} \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n\\end{table}\n")

    print(f"\nLaTeX table written to: {tex_path}")

    # Log results
    log_file = "results_log.json"
    log_data = []
    if os.path.exists(log_file):
        with open(log_file) as f:
            log_data = json.load(f)
    log_data.append({"run_tag": "exp_e_early_window_sensitivity", "results": results})
    with open(log_file, "w") as f:
        json.dump(log_data, f, indent=2)

    print("EXP-E completed successfully.")

if __name__ == "__main__":
    main()
