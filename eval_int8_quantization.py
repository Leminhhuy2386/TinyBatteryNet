#!/usr/bin/env python3
"""
eval_int8_quantization.py — EXP-D: INT8 Quantization Evaluation

Quantizes TinyBatteryNet model weights/activations to INT8 (post-training quantization)
and measures accuracy (MAPE, Acc@15%) and memory footprint (KB) across all 4 domains.

Responds to: R7 comment on INT8 quantization impact.
Output: tables/table_int8_quantization.tex
        results_log.json
"""
import os, sys, json, types, importlib, tempfile
from pathlib import Path
import joblib, numpy as np, torch
import torch.nn as nn

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

def get_model_size_kb(model):
    with tempfile.NamedTemporaryFile(suffix=".pt") as f:
        torch.save(model.state_dict(), f.name)
        return os.path.getsize(f.name) / 1024.0

def load_fp32_model(ckpt_dir):
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

def eval_model(model, args, ls, lcs):
    test_data, test_loader = data_provider_baseline(args, "test", None, ls, life_class_scaler=lcs)
    std_val = float(np.sqrt(test_data.label_scaler.var_[-1]))
    mean_val = float(test_data.label_scaler.mean_[-1])
    preds, refs, seen_unseen = [], [], []
    with torch.no_grad():
        for batch in test_loader:
            ccd, cam, lbl, _lc, _slc, _w, su_ids = batch
            ccd = ccd.float()
            cam = cam.float()
            out = model(ccd, cam)
            p = (out * std_val + mean_val).cpu().numpy().reshape(-1)
            r = (lbl * std_val + mean_val).numpy().reshape(-1)
            preds.extend(p.tolist())
            refs.extend(r.tolist())
            seen_unseen.extend(su_ids.numpy().reshape(-1).tolist())
    return compute_metrics(preds, refs, seen_unseen)

def main():
    print("=" * 60)
    print("EXP-D: INT8 Dynamic Quantization Evaluation")
    print("=" * 60)

    results = {}
    for domain in DOMAINS:
        name = PAPER_CKPT[domain]
        ckpt_dir = Path(PAPER_BUNDLE_DIR) / name
        if not ckpt_dir.exists():
            ckpt_dir = Path(CHECKPOINTS) / name
        if not ckpt_dir.exists():
            print(f"Skipping {domain}: checkpoint not found")
            continue

        print(f"\nEvaluating {domain}...")
        fp32_model, args, ls, lcs = load_fp32_model(ckpt_dir)
        fp32_size = get_model_size_kb(fp32_model)
        fp32_metrics = eval_model(fp32_model, args, ls, lcs)

        # Dynamic quantization targeting Linear and GRU layers
        int8_model = torch.ao.quantization.quantize_dynamic(
            fp32_model, {nn.Linear, nn.GRU}, dtype=torch.qint8
        )
        int8_size = get_model_size_kb(int8_model)
        int8_metrics = eval_model(int8_model, args, ls, lcs)

        results[domain] = {
            "fp32_mape": fp32_metrics["mape"],
            "fp32_acc1": fp32_metrics["acc1"],
            "fp32_size_kb": fp32_size,
            "int8_mape": int8_metrics["mape"],
            "int8_acc1": int8_metrics["acc1"],
            "int8_size_kb": int8_size,
            "mape_diff": int8_metrics["mape"] - fp32_metrics["mape"],
            "size_reduction": (1 - int8_size / fp32_size) * 100,
        }

        print(f"  FP32: MAPE = {fp32_metrics['mape']*100:.2f}%, Acc@15% = {fp32_metrics['acc1']*100:.2f}%, Size = {fp32_size:.1f} KB")
        print(f"  INT8: MAPE = {int8_metrics['mape']*100:.2f}%, Acc@15% = {int8_metrics['acc1']*100:.2f}%, Size = {int8_size:.1f} KB")
        print(f"  Delta MAPE: {results[domain]['mape_diff']*100:+.2f}% | Size Reduction: {results[domain]['size_reduction']:.1f}%")

    os.makedirs("tables", exist_ok=True)
    tex_path = "tables/table_int8_quantization.tex"
    with open(tex_path, "w") as f:
        f.write("% INT8 Quantization Evaluation Table\n")
        f.write("\\begin{table}[htbp]\n\\centering\n")
        f.write("\\caption{Impact of post-training INT8 dynamic quantization on prediction accuracy and memory footprint across chemistry domains.}\n")
        f.write("\\label{tab:int8_quantization}\n")
        f.write("\\begin{tabular}{l ccc ccc}\n\\toprule\n")
        f.write("Domain & \\multicolumn{3}{c}{FP32 Baseline} & \\multicolumn{3}{c}{INT8 Quantized} \\\\\n")
        f.write("\\cmidrule(lr){2-4} \\cmidrule(lr){5-7}\n")
        f.write(" & Size (KB) & MAPE (\\%) & Acc@15\\% & Size (KB) & MAPE (\\%) & Acc@15\\% \\\\\n\\midrule\n")
        for dom in DOMAINS:
            if dom in results:
                r = results[dom]
                f.write(f"{dom} & {r['fp32_size_kb']:.1f} & {r['fp32_mape']*100:.2f} & {r['fp32_acc1']*100:.1f}\\% & "
                        f"{r['int8_size_kb']:.1f} & {r['int8_mape']*100:.2f} & {r['int8_acc1']*100:.1f}\\% \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n\\end{table}\n")

    print(f"\nLaTeX table written to: {tex_path}")

    # Log results
    log_file = "results_log.json"
    log_data = []
    if os.path.exists(log_file):
        with open(log_file) as f:
            log_data = json.load(f)
    log_data.append({"run_tag": "exp_d_int8_quantization", "results": results})
    with open(log_file, "w") as f:
        json.dump(log_data, f, indent=2)

    print("EXP-D completed successfully.")

if __name__ == "__main__":
    main()
