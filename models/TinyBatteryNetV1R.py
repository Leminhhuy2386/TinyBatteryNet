"""
TinyBatteryNet v1R — Refined hyperparameters, same architecture as v1.

This file is a pure alias: it re-exports the Model class from TinyBatteryNetV1
so the trainer can load it under the name 'TinyBatteryNetV1R'.

Per-dataset hyperparameter overrides live in:
  exp_configs/TinyBatteryNetV1R.json            (base / shared defaults)
  exp_configs/TinyBatteryNetV1R_MIX_large.json  (Li-ion tuning)
  exp_configs/TinyBatteryNetV1R_ZN-coin.json    (Zn-ion tuning — largest gap)
  exp_configs/TinyBatteryNetV1R_NA-ion.json     (Na-ion tuning — already good)
  exp_configs/TinyBatteryNetV1R_CALB.json       (CALB tuning)
"""
from models.TinyBatteryNetV1 import Model  # noqa: F401

__all__ = ['Model']
