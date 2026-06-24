# TinyBatteryNet: Microcontroller-Deployable Deep Learning for Battery RUL Prediction

**Full Code and Results will be public after the manuscript is accepted for publication**

Welcome to the official repository for **TinyBatteryNet**, a family of micro-sized, microcontroller-deployable deep learning architectures designed for real-time Remaining Useful Life (RUL) prediction on edge devices (like Battery Management Systems - BMS).

TinyBatteryNet achieves state-of-the-art results while keeping parameter counts and memory footprint tiny enough to run on low-cost hardware.

---

## ⚡ Highlights

*   **Microcontroller-Friendly Footprint**: With only **~43 K parameters** and a **~170 KB** FP32 size (**~43 KB** INT8 quantized), TinyBatteryNet fits easily within the flash/RAM budget of STM32-class devices (e.g., 512 KB flash, 128 KB RAM).
*   **Real-time Edge Inference**: Run INT8-quantized inference in roughly **0.5 ms** on a 168 MHz STM32 microcontroller.
*   **Domain-Wide Superiority**: TinyBatteryNet beats the domain winners from the ACM KDD 2025 *BatteryLife* benchmark across all 4 battery domains (**Li-ion**, **Zn-ion**, **Na-ion**, and **CALB**).

---

## 📊 Results Summary

The table below compares the best-performing configuration of **TinyBatteryNet (V1R)** against the baseline winners from the **BatteryLife** paper benchmark (measured in Mean Absolute Percentage Error (MAPE) and Accuracy @ 15% error tolerance):

| Domain | Best Paper Baseline | Paper Best MAPE | TinyBatteryNet (V1R) MAPE ↓ | TinyBatteryNet (V1R) Acc@15% ↑ | Improvement (ΔMAPE) |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Li-ion** (MIX_large) | CPMLP | 0.179 | **0.164** | **63.0%** | **+0.015** |
| **Zn-ion** (ZN-coin) | CPTransformer | 0.515 | **0.346** | **33.0%** | **+0.169** |
| **Na-ion** (NA-ion) | CPTransformer | 0.255 | **0.232** | **40.0%** | **+0.023** |
| **CALB** (CALB) | CPMLP | 0.140 | **0.123** | **67.8%** | **+0.017** |

> [!NOTE]
> **TinyBatteryNet beats the paper domain winner on MAPE in all 4 domains, with the largest improvement in the Zn-ion domain (+0.169 lower MAPE).**

---

## 🛠️ Architecture & Inductive Biases

TinyBatteryNet's high performance at such a small scale is driven by three targeted inductive biases:

1.  **Multi-Scale Depthwise-Separable Pyramid**: Uses pyramid kernel sizes ($k = 15, 31, 61$) to capture short-term noise, mid-scale charge-plateau features, and long-range degradation trends simultaneously.
2.  **Squeeze-and-Excitation (SE) Channel Gating**: Dynamically scales and re-weights the input features (voltage, current, capacity) on a per-channel basis to automatically adapt to different battery chemistries.
3.  **Learnable Cycle Gate**: Employs a $\sigma(Wx) \times \text{mask}$ gate to suppress padding/missing cycles before feeding temporal features to the GRU, preventing gradient corruption.
4.  **Single-layer Temporal GRU**: Summarizes temporal degradation over up to 100 early-life cycles with minimal parameters.

### Model Efficiency Comparison

| Model              | Parameters |  FP32 Size  | INT8 Size (Quantized) |  RAM Usage  | Estimated STM32 Latency |
| :----------------- | :--------: | :---------: | :-------------------: | :---------: | :---------------------: |
| **TinyBatteryNet** |  **~43 K** | **~170 KB** |       **~42 KB**      | **~339 KB** |       **~0.3 ms**       |
| CPMLP              |   ~2.15 M  |   ~8.4 MB   |        ~2.1 MB        |   ~16.8 MB  |         ~12.8 ms        |
| CPTransformer      |   ~1.06 M  |   ~4.1 MB   |        ~1.0 MB        |   ~8.2 MB   |         ~6.3 ms         |
| CPGRU              |   ~1.99 M  |   ~7.8 MB   |        ~1.95 MB       |   ~15.6 MB  |         ~11.9 ms        |
| iTransformer       |   ~4.64 M  |   ~18.1 MB  |        ~4.5 MB        |   ~36.2 MB  |         ~27.6 ms        |
| MLP                |   ~5.76 M  |   ~22.5 MB  |        ~5.6 MB        |   ~45.0 MB  |         ~34.3 ms        |

---

#### Key Observations

* **TinyBatteryNet** is by far the most efficient model in terms of parameter count, memory footprint, and inference latency.
* Compared with larger baseline models, TinyBatteryNet achieves:

  * **25–130× fewer parameters**
  * **24–132× lower storage cost**
  * **20–114× faster inference on STM32F4**
* This makes TinyBatteryNet highly suitable for **resource-constrained edge deployment** and **embedded battery monitoring systems**.

---

## 📈 Detailed Training & Validation Logs

A complete breakdown of validation scores, ablation studies, and training logs is available in the [results.md](file:///work/huy.leminh/code/Dr.Huy/AI%20Battery/TinyBatteryNet/results.md) file in this repository.

To run evaluation on pre-trained checkpoints:
```bash
python evaluate_model.py
```
Or to analyze inference caching:
```bash
python main.py
```

---

### Reference & Citation

If you use the datasets or benchmark code, please cite the parent benchmark:
```bibtex
@inproceedings{10.1145/3711896.3737372,
  author = {Tan, Ruifeng and Hong, Weixiang and Tang, Jiayue and Lu, Xibin and Ma, Ruijun and Zheng, Xiang and Li, Jia , Huang, Jiaqiang and Zhang, Tong-Yi},
  title = {BatteryLife: A Comprehensive Dataset and Benchmark for Battery Life Prediction},
  year = {2025},
  booktitle = {Proceedings of the 31st ACM SIGKDD Conference on Knowledge Discovery and Data Mining V.2},
  pages = {5789–5800},
  doi = {10.1145/3711896.3737372}
}
```
