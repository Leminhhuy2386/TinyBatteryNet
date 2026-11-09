#!/usr/bin/env python3
"""
eval_fixed_discard_rule.py — EXP-G: Learnable Cycle Gate vs. Fixed 5-Cycle Discard Rule

Quantitatively compares:
  1. Learnable Cycle Gate (Ours - Full TinyBatteryNet)
  2. Fixed 5-Cycle Discard Rule (Heuristic: zero out first 5 cycles, no gate)
  3. Fixed 12-Cycle Discard Rule (Heuristic: zero out first 12 cycles, no gate)
  4. No Gate / No Discard Baseline (Unweighted pass-through of all cycles)

Responds directly to Reviewer #2 Comment 1.
Output: tables/table_fixed_discard_vs_learnable_gate.tex
        figures data/table_fixed_discard_vs_learnable_gate.csv
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

NO_GATE_CKPT = {
    "Li-ion": "TinyBatteryNetV1R_NoGate_sl1_lr0.0003_dm64_nh4_el2_dl1_df16_lradjplateau_datasetMIX_large_lossMAPE_wd0.0001_wlTrue_bs32_s42-TinyBatteryNetV1R_NoGate_ablation_no_gate",
    "Zn-ion": "TinyBatteryNetV1R_NoGate_sl1_lr0.0001_dm64_nh4_el2_dl1_df16_lradjplateau_datasetZN-coin_lossMAPE_wd0.003_wlTrue_bs32_s42-TinyBatteryNetV1R_NoGate_ablation_no_gate",
    "Na-ion": "TinyBatteryNetV1R_NoGate_sl1_lr0.0003_dm64_nh4_el2_dl1_df16_lradjplateau_datasetNA-ion_lossMAPE_wd0.0001_wlTrue_bs16_s42-TinyBatteryNetV1R_NoGate_ablation_no_gate",
    "CALB":   "TinyBatteryNetV1R_NoGate_sl1_lr0.00045_dm64_nh4_el2_dl1_df16_lradjconstant_datasetCALB_lossMSE_wd0.0005_wlFalse_bs16_s42-TinyBatteryNetV1R_NoGate_ablation_no_gate",
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

def eval_discard_rule(model, args, ls, lcs, discard_cycles=0):
    test_data, test_loader = data_provider_baseline(args, "test", None, ls, life_class_scaler=lcs)
    std_val = float(np.sqrt(test_data.label_scaler.var_[-1]))
    mean_val = float(test_data.label_scaler.mean_[-1])
    preds, refs, seen_unseen = [], [], []
    with torch.no_grad():
        for batch in test_loader:
            ccd, cam, lbl, _lc, _slc, _w, su_ids = batch
            ccd = ccd.float()
            cam = cam.float()
            # Manually discard early cycles by setting attention mask to zero for cycles < discard_cycles
            if discard_cycles > 0:
                cam[:, :discard_cycles] = 0.0

            out = model(ccd, cam)
            p = (out * std_val + mean_val).cpu().numpy().reshape(-1)
            r = (lbl * std_val + mean_val).numpy().reshape(-1)
            preds.extend(p.tolist())
            refs.extend(r.tolist())
            seen_unseen.extend(su_ids.numpy().reshape(-1).tolist())
    return compute_metrics(preds, refs, seen_unseen)

def main():
    print("=" * 70)
    print("EXP-G: Learnable Cycle Gate vs. Fixed Discard Rule Comparison")
    print("=" * 70)

    results = {}
    for domain in DOMAINS:
        print(f"\nEvaluating {domain}...")
        results[domain] = {}

        # 1. Learnable Cycle Gate (Ours)
        ckpt_gate = Path(PAPER_BUNDLE_DIR) / PAPER_CKPT[domain]
        if not ckpt_gate.exists(): ckpt_gate = Path(CHECKPOINTS) / PAPER_CKPT[domain]
        if ckpt_gate.exists():
            model, args, ls, lcs = load_model(ckpt_gate)
            m_learnable = eval_discard_rule(model, args, ls, lcs, discard_cycles=0)
            results[domain]["Learnable_Gate"] = m_learnable
            print(f"  Learnable Cycle Gate (Ours) : MAPE = {m_learnable['mape']*100:.2f}% | Acc@15% = {m_learnable['acc1']*100:.2f}%")

        # 2. No Gate Baselines with Fixed Discard Rules
        ckpt_nogate = Path(PAPER_BUNDLE_DIR) / NO_GATE_CKPT[domain]
        if not ckpt_nogate.exists(): ckpt_nogate = Path(CHECKPOINTS) / NO_GATE_CKPT[domain]
        if ckpt_nogate.exists():
            model_ng, args_ng, ls_ng, lcs_ng = load_model(ckpt_nogate)
            
            # (a) No Discard (all cycles unweighted)
            m_nogate = eval_discard_rule(model_ng, args_ng, ls_ng, lcs_ng, discard_cycles=0)
            results[domain]["No_Gate_No_Discard"] = m_nogate
            print(f"  No Gate / All Cycles       : MAPE = {m_nogate['mape']*100:.2f}% | Acc@15% = {m_nogate['acc1']*100:.2f}%")

            # (b) Fixed 5-Cycle Discard Rule (CyclePatch rule)
            m_fixed5 = eval_discard_rule(model_ng, args_ng, ls_ng, lcs_ng, discard_cycles=5)
            results[domain]["Fixed_5_Cycle_Discard"] = m_fixed5
            print(f"  Fixed 5-Cycle Discard      : MAPE = {m_fixed5['mape']*100:.2f}% | Acc@15% = {m_fixed5['acc1']*100:.2f}%")

            # (c) Fixed 12-Cycle Discard Rule
            m_fixed12 = eval_discard_rule(model_ng, args_ng, ls_ng, lcs_ng, discard_cycles=12)
            results[domain]["Fixed_12_Cycle_Discard"] = m_fixed12
            print(f"  Fixed 12-Cycle Discard     : MAPE = {m_fixed12['mape']*100:.2f}% | Acc@15% = {m_fixed12['acc1']*100:.2f}%")

    # Export LaTeX table
    os.makedirs("tables", exist_ok=True)
    tex_path = "tables/table_fixed_discard_vs_learnable_gate.tex"
    with open(tex_path, "w") as f:
        f.write("% Fixed Discard Rule vs. Learnable Cycle Gate Comparison Table\n")
        f.write("\\begin{table}[htbp]\n\\centering\n")
        f.write("\\caption{Quantitative comparison of prediction MAPE (\\%) and Acc@15\\% between heuristic fixed-cycle discard rules (CyclePatch approach) and our learnable Cycle Gate across battery chemistry domains.}\n")
        f.write("\\label{tab:fixed_discard_vs_learnable_gate}\n")
        f.write("\\begin{tabular}{l cc cc cc cc}\n\\toprule\n")
        f.write("Strategy & \\multicolumn{2}{c}{Li-ion} & \\multicolumn{2}{c}{Zn-ion} & \\multicolumn{2}{c}{Na-ion} & \\multicolumn{2}{c}{CALB} \\\\\n")
        f.write("\\cmidrule(lr){2-3} \\cmidrule(lr){4-5} \\cmidrule(lr){6-7} \\cmidrule(lr){8-9}\n")
        f.write(" & MAPE (\\%) & Acc@15\\% & MAPE (\\%) & Acc@15\\% & MAPE (\\%) & Acc@15\\% & MAPE (\\%) & Acc@15\\% \\\\\n\\midrule\n")

        for key, name in [
            ("No_Gate_No_Discard", "No Gate (All 100 cycles)"),
            ("Fixed_5_Cycle_Discard", "Fixed 5-Cycle Discard (CyclePatch)"),
            ("Fixed_12_Cycle_Discard", "Fixed 12-Cycle Discard"),
            ("Learnable_Gate", "\\textbf{Learnable Cycle Gate (Ours)}"),
        ]:
            row_str = name
            for dom in DOMAINS:
                if dom in results and key in results[dom]:
                    res = results[dom][key]
                    row_str += f" & {res['mape']*100:.2f} & {res['acc1']*100:.1f}\\%"
                else:
                    row_str += " & N/A & N/A"
            row_str += " \\\\\n"
            f.write(row_str)
        f.write("\\bottomrule\n\\end{tabular}\n\\end{table}\n")

    print(f"\nLaTeX table written to: {tex_path}")

    # Export CSV data to figures data/
    csv_rows = []
    for dom in DOMAINS:
        if dom in results:
            for strat, res in results[dom].items():
                csv_rows.append({
                    "domain": dom,
                    "strategy": strat,
                    "test_mape": res["mape"],
                    "test_acc15": res["acc1"],
                    "test_mae": res["mae"]
                })
    import pandas as pd
    df = pd.DataFrame(csv_rows)
    csv_path = "figures data/table_fixed_discard_vs_learnable_gate.csv"
    df.to_csv(csv_path, index=False)
    print(f"CSV data saved to: {csv_path}")

if __name__ == "__main__":
    main()
