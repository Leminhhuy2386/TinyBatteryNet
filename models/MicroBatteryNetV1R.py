"""
MicroBatteryNet v1R — Refined hyperparameters, same architecture as v1.

This file is a pure alias: it re-exports the Model class from MicroBatteryNetV1
so the trainer can load it under the name 'MicroBatteryNetV1R'.

Per-dataset hyperparameter overrides live in:
  exp_configs/MicroBatteryNetV1R.json            (base / shared defaults)
  exp_configs/MicroBatteryNetV1R_MIX_large.json  (Li-ion tuning)
  exp_configs/MicroBatteryNetV1R_ZN-coin.json    (Zn-ion tuning — largest gap)
  exp_configs/MicroBatteryNetV1R_NA-ion.json     (Na-ion tuning — already good)
  exp_configs/MicroBatteryNetV1R_CALB.json       (CALB tuning)
"""
from models.MicroBatteryNetV1 import Model  # noqa: F401

__all__ = ['Model']
