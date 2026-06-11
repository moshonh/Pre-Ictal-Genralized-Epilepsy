# IID Analyzer — Pre-ictal Discharge Analysis

**Rambam Epilepsy Unit | Herskovitz Lab**

Streamlit web app for automated analysis of interictal discharges (IIDs)
in generalized epilepsy patients, based on EDF+ Video-EEG files.

---

## What it does

1. **Reads EDF+** with embedded EDF Annotations channel
2. **Auto-detects seizure onset** from annotation keywords (seizure, sz, TC, GTC, onset, התקף)
3. **Detects IIDs** using threshold-based peak detection on the global-average signal
4. **Hourly analysis**: IID frequency and duration per hour in the N hours before seizure
5. **Spectral analysis**: relative band power (δ/θ/α/β/γ) per discharge + temporal evolution
6. **Trend classification**: ↑ Increasing (overexcitation) or ↓ Decreasing (critical slowing down)
7. **Export** to CSV for further statistical analysis

---

## Installation

```bash
pip install -r requirements.txt
```

## Run

```bash
streamlit run app.py
```

---

## Detection algorithm

- Signal: average reference across selected EEG channels (appropriate for generalized epilepsy)
- Filter: bandpass 1–70 Hz + 50 Hz notch
- Detection: MAD-based Z-score, peak finding with refractory period
- Discharge extent: 30% of peak Z as boundary criterion
- Duration filter: configurable min/max (default 20–250 ms)

## Key parameters (sidebar)

| Parameter | Default | Notes |
|---|---|---|
| Z-score threshold | 4.0 | Lower = more sensitive, more false positives |
| Min spike duration | 20 ms | Exclude artifacts |
| Max spike duration | 250 ms | Exclude slow waves / artifacts |
| Refractory period | 500 ms | Min inter-discharge interval |
| Hours before seizure | 8 | Analysis window |

---

## Biological rationale

In generalized epilepsy, IIDs are synchronous across the scalp — the scalp EEG
reflects the same network dynamics visible at the single-neuron level.
Two pre-ictal patterns are expected (Herskovitz et al.):

- **Increasing IID rate** → progressive overexcitation preceding seizure
- **Decreasing IID rate** → "critical slowing down" (Maturana et al., Nat Commun 2020):
  the epileptogenic network approaches a bifurcation point, inhibition fails,
  and the seizure emerges after apparent suppression

---

## Output files

- `iid_detections.csv` — per-IID table with onset, duration, Z-score, hour bin
- `iid_spectral.csv` — per-IID spectral band powers
