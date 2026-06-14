"""
IID Analyzer — Interictal Discharge Analysis for Generalized Epilepsy
Analyzes pre-ictal IID frequency, duration, and spectral properties
from EDF+ files with EDF Annotations.
"""

import streamlit as st
import mne
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from scipy import signal
from scipy.stats import linregress
import io

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="IID Analyzer",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .main > div { padding-top: 1rem; }
    .stMetric label { font-size: 0.85rem; }
    .stAlert { font-size: 0.9rem; }
    h1 { font-size: 1.6rem; }
    h2 { font-size: 1.2rem; }
    h3 { font-size: 1rem; }
</style>
""", unsafe_allow_html=True)


# ── Helpers ───────────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner=False)
def load_edf(_file_bytes: bytes):
    """Load EDF from bytes, return raw MNE object."""
    with io.BytesIO(_file_bytes) as buf:
        raw = mne.io.read_raw_edf(buf, preload=True, verbose=False)
    return raw


def get_seizure_onset(raw, custom_keyword: str = "") -> float | None:
    """
    Extract seizure onset in seconds from recording start.
    Handles both standard EDF+ (onset relative) and Natus/Xltek style
    where onset=0 and the absolute time is encoded in orig_time.
    """
    import datetime

    sz_keywords = ["seizure", "sz", "tc", "gtc", "ictal", "התקף"]
    if custom_keyword:
        sz_keywords = [custom_keyword.lower()]

    rec_start = raw.info.get("meas_date", None)

    candidates = []
    for ann in raw.annotations:
        desc = ann["description"].lower().strip()

        if custom_keyword:
            if custom_keyword.lower() not in desc:
                continue
        else:
            is_sz = any(k in desc for k in sz_keywords)
            is_onset = desc in ("onset", "seizure onset")
            if not (is_sz or is_onset):
                continue

        onset_rel = float(ann["onset"])

        # onset already relative to recording start
        if onset_rel > 0.5:
            candidates.append(onset_rel)
            continue

        # Natus/Xltek: onset=0, absolute time in orig_time
        orig_time = ann.get("orig_time", None)
        if orig_time is not None and rec_start is not None:
            try:
                if hasattr(orig_time, "tzinfo") and orig_time.tzinfo is None:
                    orig_time = orig_time.replace(tzinfo=datetime.timezone.utc)
                rs = rec_start
                if hasattr(rs, "tzinfo") and rs.tzinfo is None:
                    rs = rs.replace(tzinfo=datetime.timezone.utc)
                delta = (orig_time - rs).total_seconds()
                if delta >= 0:
                    candidates.append(delta)
                    continue
            except Exception:
                pass

        candidates.append(onset_rel)

    if not candidates:
        return None

    return max(candidates)


def bandpass(data, sfreq, lo=1.0, hi=70.0, notch=50.0):
    """Bandpass + notch filter."""
    b, a = signal.butter(4, [lo / (sfreq / 2), hi / (sfreq / 2)], btype="band")
    filtered = signal.filtfilt(b, a, data, axis=-1)
    b_n, a_n = signal.iirnotch(notch, 30, sfreq)
    filtered = signal.filtfilt(b_n, a_n, filtered, axis=-1)
    return filtered


def detect_iids(
    data: np.ndarray,
    sfreq: float,
    z_thresh: float = 4.0,
    min_dur_ms: float = 20,
    max_dur_ms: float = 250,
    refractory_ms: float = 500,
) -> list[dict]:
    """
    Burst detector for generalized spike-wave discharges.

    Detects the entire burst (onset → offset) using smoothed GFP envelope.
    Each burst contains multiple spike-wave complexes.
    Returns burst onset, duration, spike power, slow-wave power, and E/I ratio.
    """
    # ── Bandpass 1-70 Hz ──
    b, a = signal.butter(4, [1.0 / (sfreq / 2), 70.0 / (sfreq / 2)], btype="band")
    data_f = signal.filtfilt(b, a, data, axis=-1)

    # ── Smoothed GFP (200ms window) ──
    gfp = np.sqrt(np.mean(data_f ** 2, axis=0))
    kernel = np.ones(int(0.2 * sfreq)) / int(0.2 * sfreq)
    gfp_s  = np.convolve(gfp, kernel, mode="same")

    # ── Threshold: median + z_thresh * MAD ──
    med = np.median(gfp_s)
    mad = np.median(np.abs(gfp_s - med)) + 1e-30
    thresh = med + z_thresh * mad

    # ── Binary burst mask → segments ──
    mask = (gfp_s > thresh).astype(int)
    diff = np.diff(mask)
    onsets  = np.where(diff == 1)[0]
    offsets = np.where(diff == -1)[0]
    if len(offsets) and len(onsets) and offsets[0] < onsets[0]:
        offsets = offsets[1:]
    n = min(len(onsets), len(offsets))
    onsets, offsets = onsets[:n], offsets[:n]

    min_samps = int(300  / 1000 * sfreq)   # min burst 300ms
    max_samps = int(30.0 * sfreq)           # max burst 30s

    _trapz = getattr(np, "trapezoid", getattr(np, "trapz", None))
    # Use RAW (unfiltered) data for PSD — bandpass would kill delta/slow-wave content
    # GFP preserves generalized synchrony without cancellation artifacts of simple average
    gfp_raw = np.sqrt(np.mean(data ** 2, axis=0))

    iids = []
    for on, off in zip(onsets, offsets):
        dur = off - on
        if dur < min_samps or dur > max_samps:
            continue

        seg = gfp_raw[on:off]
        n_seg = len(seg)

        # ── Welch PSD: 2s window for good low-freq resolution ──
        nperseg = min(n_seg, max(64, int(2.0 * sfreq)))
        freqs, psd = signal.welch(seg, fs=sfreq, nperseg=nperseg,
                                   noverlap=nperseg // 2, scaling="density")

        # ── Normalize by total power (relative PSD) ──
        total_power = _trapz(psd, freqs) + 1e-30

        def band_power(lo, hi):
            mask = (freqs >= lo) & (freqs < hi)
            return float(_trapz(psd[mask], freqs[mask]) / total_power)

        # Excitatory: spike component (15–70 Hz)
        excit = band_power(15, 70)
        # Inhibitory: slow wave (1–4 Hz)
        inhib = band_power(1, 4)
        # E/I ratio (normalized — no band-width bias)
        ei_ratio = excit / (inhib + 1e-30)

        # Full spectral profile
        delta  = band_power(1,  4)
        theta  = band_power(4,  8)
        alpha  = band_power(8,  13)
        beta   = band_power(13, 30)
        gamma  = band_power(30, 70)

        # Peak of GFP within burst
        pk = on + int(gfp[on:off].argmax())

        # ── True onset: walk back from GFP threshold crossing ('on') using
        #    the LOCAL MINIMUM GFP in the 500ms window before 'on' as reference.
        #    This avoids contamination from other bursts in the baseline window.
        #    Threshold = local_min * 1.5 (50% above local quiet level).
        #    Max lookback = 200ms. ──
        pre_start = max(0, on - int(0.5 * sfreq))
        pre_end   = on
        if pre_end > pre_start + 5:
            local_min = float(np.percentile(gfp[pre_start:pre_end], 10))
            bl_thresh = local_min * 2.0  # 2x local minimum
        else:
            bl_thresh = gfp[pk] * 0.15  # fallback

        max_lookback = int(0.2 * sfreq)  # max 200ms lookback
        true_onset = on
        for s in range(on, max(0, on - max_lookback), -1):
            if gfp[s] < bl_thresh:
                true_onset = s
                break

        iids.append({
            "onset_sample":  int(true_onset),
            "peak_sample":   int(pk),
            "offset_sample": int(off),
            "onset_sec":     true_onset / sfreq,
            "duration_ms":   (off - true_onset) / sfreq * 1000,
            "peak_z":        float(gfp[pk] / (mad + 1e-30)),
            "peak_amp":      float(gfp[pk] * 1e6),
            # Welch-based spectral measures
            "spike_power":   excit,   # relative power 15-70Hz
            "slow_power":    inhib,   # relative power 1-4Hz
            "ei_ratio":      ei_ratio,
            "delta_rel":     delta,
            "theta_rel":     theta,
            "alpha_rel":     alpha,
            "beta_rel":      beta,
            "gamma_rel":     gamma,
        })

    return iids


def compute_psd_per_iid(
    data: np.ndarray,
    sfreq: float,
    iids: list[dict],
    context_ms: float = 100,
) -> pd.DataFrame:
    """
    For each IID, compute PSD in standard EEG bands.
    Returns DataFrame with band powers per discharge.
    """
    bands = {
        "delta (1-4 Hz)":   (1,  4),
        "theta (4-8 Hz)":   (4,  8),
        "alpha (8-13 Hz)":  (8,  13),
        "beta (13-30 Hz)":  (13, 30),
        "gamma (30-70 Hz)": (30, 70),
    }
    context = int(context_ms / 1000 * sfreq)
    avg = np.mean(data, axis=0)

    rows = []
    for iid in iids:
        start = max(0, iid["onset_sample"] - context)
        end   = min(len(avg), iid["offset_sample"] + context)
        seg   = avg[start:end]

        if len(seg) < 32:
            continue

        freqs, psd = signal.welch(seg, fs=sfreq, nperseg=min(len(seg), 128))
        row = {"onset_sec": iid["onset_sec"], "duration_ms": iid["duration_ms"]}
        _trapz = getattr(np, "trapezoid", getattr(np, "trapz", None))
        total_power = _trapz(psd, freqs) + 1e-30
        for band, (lo, hi) in bands.items():
            mask = (freqs >= lo) & (freqs < hi)
            row[band] = float(_trapz(psd[mask], freqs[mask]) / total_power)
        rows.append(row)

    return pd.DataFrame(rows)


def assign_hourly_bins(iids: list[dict], seizure_onset_sec: float, hours_before: int = 8):
    """
    Assign each IID to an hour-bin relative to seizure onset.
    Bin label = hours before seizure (8 = earliest, 1 = last hour before seizure).
    """
    records = []
    for iid in iids:
        t_rel = iid["onset_sec"] - seizure_onset_sec  # negative = before seizure
        if -hours_before * 3600 <= t_rel < 0:
            hour_bin = int(np.ceil(-t_rel / 3600))  # 1..hours_before
            hour_bin = min(hour_bin, hours_before)
            records.append({**iid, "t_rel_sec": t_rel, "hour_bin": hour_bin})
    return records


def trend_label(slope: float, pval: float) -> str:
    if pval > 0.1:
        return "No clear trend"
    return "↑ Increasing" if slope > 0 else "↓ Decreasing"


# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("🧠 IID Analyzer")
    st.caption("Generalized Epilepsy | Pre-ictal Discharge Analysis")
    st.divider()

    uploaded = st.file_uploader("Upload EDF+ file", type=["edf"])

    st.subheader("⚙️ Detection parameters")
    z_thresh = st.slider("Z-score threshold", 2.0, 8.0, 4.0, 0.5,
                         help="Higher = fewer but more confident detections")
    min_dur  = st.slider("Min spike duration (ms)", 10, 60, 20, 5)
    max_dur  = st.slider("Max spike duration (ms)", 100, 500, 250, 25)
    refrac   = st.slider("Refractory period (ms)", 200, 2000, 500, 100)
    hours_before = st.slider("Hours before seizure to analyze", 2, 12, 8, 1)

    st.subheader("🏷️ Seizure annotation keyword")
    seizure_keyword = st.text_input(
        "Custom keyword (leave blank for auto-detect)",
        value="",
        help="e.g. 'sz', 'TC', 'onset'. Case-insensitive.",
    )

    st.divider()
    st.caption("Rambam Epilepsy Unit | Herskovitz Lab")


# ── Main ──────────────────────────────────────────────────────────────────────

st.title("🧠 Pre-ictal IID Analyzer")
st.caption("Interictal Discharge Frequency, Duration & Spectral Analysis — Generalized Epilepsy")

if uploaded is None:
    st.info("Upload an EDF+ file from the sidebar to begin.")
    st.markdown("""
    **What this tool does:**
    - Reads EDF+ files with embedded annotations
    - Auto-detects seizure onset from annotation channel
    - Detects interictal discharges (IIDs) using threshold-based peak detection
    - Analyzes IID **frequency** and **duration** per hour leading to seizure
    - Computes **spectral band power** per discharge
    - Classifies pre-ictal pattern: ↑ Increasing (overexcitation) or ↓ Decreasing (critical slowing down)
    """)
    st.stop()


# ── Load & process ────────────────────────────────────────────────────────────

with st.spinner("Loading EDF file…"):
    file_bytes = uploaded.read()
    raw = load_edf(file_bytes)

sfreq = raw.info["sfreq"]
ch_names = raw.ch_names
duration_hrs = raw.times[-1] / 3600

col1, col2, col3 = st.columns(3)
col1.metric("Sampling rate", f"{sfreq:.0f} Hz")
col2.metric("Channels", len(ch_names))
col3.metric("Recording duration", f"{duration_hrs:.1f} h")

# Seizure onset — unified via get_seizure_onset (handles Natus orig_time style)
sz_onset = get_seizure_onset(raw, custom_keyword=seizure_keyword)

if sz_onset is None:
    st.error(
        "⚠️ No seizure annotation found. Check annotation keywords in the sidebar, "
        "or verify that the EDF+ has an annotations channel."
    )
    with st.expander("All annotations in file"):
        for ann in raw.annotations:
            st.write(f"  {ann['onset']:.1f}s — `{ann['description']}`")
    st.stop()

st.success(f"✅ Seizure onset detected at **{sz_onset:.1f} s** ({sz_onset/3600:.2f} h from recording start)")

# Channel selection
st.subheader("Channel selection")
eeg_channels = [c for c in ch_names if "EEG" in c.upper() or "FP" in c.upper()
                or any(x in c.upper() for x in ["FZ","CZ","PZ","OZ","F3","F4","C3","C4","P3","P4"])]
if not eeg_channels:
    eeg_channels = ch_names[:min(19, len(ch_names))]

selected_chs = st.multiselect(
    "Select EEG channels for analysis",
    options=ch_names,
    default=eeg_channels[:min(19, len(eeg_channels))],
)

if not selected_chs:
    st.warning("Select at least one channel.")
    st.stop()


# Extract data window directly (avoid deepcopy of cached resource)
analysis_start = max(0, sz_onset - hours_before * 3600)
start_samp = int(analysis_start * sfreq)
end_samp   = int((sz_onset - 0.1) * sfreq)
end_samp   = min(end_samp, raw._data.shape[1] - 1)
ch_indices = [raw.ch_names.index(c) for c in selected_chs]
data = raw._data[ch_indices, start_samp:end_samp]

with st.spinner("Filtering & detecting IIDs…"):
    data_filt = bandpass(data, sfreq)
    iids_raw  = detect_iids(data_filt, sfreq, z_thresh, min_dur, max_dur, refrac)

# Adjust IID times to absolute recording time
offset_sec = analysis_start
for iid in iids_raw:
    iid["onset_sec"] += offset_sec

binned = assign_hourly_bins(iids_raw, sz_onset, hours_before)

if not binned:
    st.error("No IIDs detected in the pre-ictal window. Try lowering the Z-score threshold.")
    st.stop()

df_iid  = pd.DataFrame(binned)
df_psd  = compute_psd_per_iid(data_filt, sfreq, iids_raw)


# ── Summary metrics ───────────────────────────────────────────────────────────

st.divider()
st.subheader("📊 Summary")

total_iids = len(df_iid)
mean_dur   = df_iid["duration_ms"].mean()
median_dur = df_iid["duration_ms"].median()
rate_per_hr = total_iids / hours_before

c1, c2, c3, c4 = st.columns(4)
c1.metric("Total IIDs detected", total_iids)
c2.metric("Mean IID duration", f"{mean_dur:.0f} ms")
c3.metric("Median IID duration", f"{median_dur:.0f} ms")
c4.metric("Mean rate", f"{rate_per_hr:.1f} / hr")


# ── Hourly analysis ───────────────────────────────────────────────────────────

hourly = (
    df_iid.groupby("hour_bin")
    .agg(count=("onset_sec", "count"), mean_dur=("duration_ms", "mean"))
    .reset_index()
    .sort_values("hour_bin", ascending=False)  # hour 8 → hour 1
)
hourly["x_label"] = hourly["hour_bin"].apply(lambda h: f"H-{h}")

# Trend
bins_sorted = hourly.sort_values("hour_bin")
slope_freq, intercept_freq, r_freq, p_freq, _ = linregress(
    bins_sorted["hour_bin"].values,
    bins_sorted["count"].values,
)
slope_dur, intercept_dur, r_dur, p_dur, _ = linregress(
    bins_sorted["hour_bin"].values,
    bins_sorted["mean_dur"].values,
)

st.divider()
st.subheader("📈 Hourly IID Frequency & Duration")

# Trend summary
t_freq = trend_label(-slope_freq, p_freq)   # negate: hour_bin counts down to seizure
t_dur  = trend_label(-slope_dur, p_dur)

col_a, col_b = st.columns(2)
color_freq = "green" if "↑" in t_freq else ("red" if "↓" in t_freq else "gray")
color_dur  = "green" if "↑" in t_dur  else ("red" if "↓" in t_dur  else "gray")

col_a.markdown(
    f"**Frequency trend toward seizure:** <span style='color:{color_freq};font-size:1.1rem'>{t_freq}</span> "
    f"(r={r_freq:.2f}, p={p_freq:.3f})",
    unsafe_allow_html=True,
)
col_b.markdown(
    f"**Duration trend toward seizure:** <span style='color:{color_dur};font-size:1.1rem'>{t_dur}</span> "
    f"(r={r_dur:.2f}, p={p_dur:.3f})",
    unsafe_allow_html=True,
)

# Plots
fig_hourly = make_subplots(
    rows=1, cols=2,
    subplot_titles=("IID Count per Hour (H-8 → H-1)", "Mean IID Duration per Hour (ms)"),
)

x_labels = hourly.sort_values("hour_bin", ascending=False)["x_label"].tolist()
y_count  = hourly.sort_values("hour_bin", ascending=False)["count"].tolist()
y_dur    = hourly.sort_values("hour_bin", ascending=False)["mean_dur"].tolist()

fig_hourly.add_trace(
    go.Bar(x=x_labels, y=y_count, marker_color="#3B82F6", name="Count"),
    row=1, col=1,
)
fig_hourly.add_trace(
    go.Bar(x=x_labels, y=y_dur, marker_color="#10B981", name="Mean duration (ms)"),
    row=1, col=2,
)

fig_hourly.update_layout(height=380, showlegend=False, margin=dict(t=40, b=20))
fig_hourly.update_xaxes(title_text="Hour before seizure")
fig_hourly.update_yaxes(title_text="Count", row=1, col=1)
fig_hourly.update_yaxes(title_text="Duration (ms)", row=1, col=2)
st.plotly_chart(fig_hourly, use_container_width=True)


# ── IID scatter timeline ──────────────────────────────────────────────────────

st.subheader("⏱️ IID Timeline")

fig_scatter = go.Figure()
fig_scatter.add_trace(go.Scatter(
    x=df_iid["t_rel_sec"] / 3600,
    y=df_iid["duration_ms"],
    mode="markers",
    marker=dict(
        color=df_iid["peak_z"],
        colorscale="Viridis",
        size=6,
        colorbar=dict(title="Z-score"),
        showscale=True,
    ),
    hovertemplate="<b>%{x:.2f} h before seizure</b><br>Duration: %{y:.0f} ms<br>Z: %{marker.color:.1f}<extra></extra>",
))
fig_scatter.add_vline(x=0, line_color="red", line_dash="dash", annotation_text="Seizure onset")
fig_scatter.update_layout(
    xaxis_title="Hours before seizure",
    yaxis_title="IID duration (ms)",
    height=320,
    margin=dict(t=20, b=20),
)
st.plotly_chart(fig_scatter, use_container_width=True)


# ── Boxplot per hour ──────────────────────────────────────────────────────────

st.subheader("📦 IID Duration Distribution per Hour")

fig_box = go.Figure()
for h in sorted(df_iid["hour_bin"].unique(), reverse=True):
    vals = df_iid.loc[df_iid["hour_bin"] == h, "duration_ms"].values
    fig_box.add_trace(go.Box(
        y=vals,
        name=f"H-{h}",
        boxmean=True,
        marker_color="#6366F1",
    ))

fig_box.update_layout(
    xaxis_title="Hour before seizure",
    yaxis_title="IID duration (ms)",
    showlegend=False,
    height=350,
    margin=dict(t=20, b=20),
)
st.plotly_chart(fig_box, use_container_width=True)


# ── Spectral analysis ─────────────────────────────────────────────────────────

if not df_psd.empty:
    st.divider()
    st.subheader("🌊 Spectral Band Power per IID")

    band_cols = [c for c in df_psd.columns if "Hz" in c]

    # Mean band power
    band_means = df_psd[band_cols].mean()
    fig_bands = go.Figure(go.Bar(
        x=[b.split(" ")[0] for b in band_cols],
        y=band_means.values,
        marker_color=px.colors.qualitative.Plotly,
        text=[f"{v:.1%}" for v in band_means.values],
        textposition="outside",
    ))
    fig_bands.update_layout(
        title="Mean Relative Band Power across All IIDs",
        yaxis_title="Relative power",
        height=320,
        margin=dict(t=40, b=20),
    )
    st.plotly_chart(fig_bands, use_container_width=True)

    # Temporal evolution of band power
    df_psd_plot = df_psd.copy()
    df_psd_plot["t_rel_h"] = (df_psd_plot["onset_sec"] - sz_onset) / 3600
    df_psd_plot = df_psd_plot[df_psd_plot["t_rel_h"] >= -hours_before]

    fig_spec_time = go.Figure()
    colors_spec = px.colors.qualitative.Plotly
    for i, band in enumerate(band_cols):
        fig_spec_time.add_trace(go.Scatter(
            x=df_psd_plot["t_rel_h"],
            y=df_psd_plot[band].rolling(5, min_periods=1).mean(),
            mode="lines",
            name=band.split(" ")[0],
            line=dict(color=colors_spec[i % len(colors_spec)]),
        ))

    fig_spec_time.add_vline(x=0, line_color="red", line_dash="dash", annotation_text="Seizure")
    fig_spec_time.update_layout(
        title="Spectral Band Power Evolution Toward Seizure (5-IID rolling mean)",
        xaxis_title="Hours before seizure",
        yaxis_title="Relative power",
        height=380,
        margin=dict(t=40, b=20),
    )
    st.plotly_chart(fig_spec_time, use_container_width=True)

    # Per-hour spectral heatmap
    st.subheader("🗺️ Spectral Heatmap (Band × Hour)")
    df_psd["hour_bin"] = df_psd["onset_sec"].apply(
        lambda s: int(np.ceil((sz_onset - s) / 3600))
    )
    df_psd["hour_bin"] = df_psd["hour_bin"].clip(1, hours_before)
    heatmap_data = df_psd.groupby("hour_bin")[band_cols].mean()
    heatmap_data = heatmap_data.reindex(sorted(heatmap_data.index, reverse=True))

    fig_heat = go.Figure(go.Heatmap(
        z=heatmap_data.values,
        x=[b.split(" ")[0] for b in band_cols],
        y=[f"H-{h}" for h in heatmap_data.index],
        colorscale="RdBu_r",
        zmid=heatmap_data.values.mean(),
        colorbar=dict(title="Rel. power"),
    ))
    fig_heat.update_layout(
        xaxis_title="Band",
        yaxis_title="Hour before seizure",
        height=max(300, 50 * len(heatmap_data)),
        margin=dict(t=20, b=20),
    )
    st.plotly_chart(fig_heat, use_container_width=True)




# ── Research Analysis: Burst Rate / Duration / E/I Ratio ─────────────────────

st.divider()
st.subheader("🔬 Pre-ictal Burst Analysis")
st.caption("שלושת המרכיבים המחקריים: תדירות · משך · יחס עירור/עיכוב")

if len(df_iid) < 3:
    st.warning("Not enough bursts detected for trend analysis. Try lowering the Z-score threshold.")
else:
    # ── Add E/I columns if present ──
    has_ei = "ei_ratio" in df_iid.columns

    # ── Time axis: minutes before seizure ──
    df_iid["t_rel_min"] = (df_iid["onset_sec"] - sz_onset) / 60

    # ── Bin into 1-minute windows ──
    df_iid["min_bin"] = df_iid["t_rel_min"].apply(lambda t: int(np.floor(t)))
    min_bins = sorted(df_iid["min_bin"].unique())

    bin_stats = []
    for mb in min_bins:
        sub = df_iid[df_iid["min_bin"] == mb]
        row = {
            "min_before_sz": -mb,
            "t_rel_min": mb,
            "burst_count": len(sub),
            "mean_dur_s": sub["duration_ms"].mean() / 1000,
        }
        if has_ei:
            row["mean_ei"] = sub["ei_ratio"].median()
        bin_stats.append(row)

    df_bins = pd.DataFrame(bin_stats).sort_values("t_rel_min")

    from scipy.stats import linregress

    # ── Plot 1: Burst Rate ──
    st.markdown("#### 1 · Burst Rate (bursts/min)")
    slope_r, _, r_r, p_r, _ = linregress(df_bins["t_rel_min"], df_bins["burst_count"])
    trend_r = ("↑ Increasing" if slope_r < 0 and p_r < 0.1 else
               "↓ Decreasing" if slope_r > 0 and p_r < 0.1 else "No clear trend")
    st.markdown(f"**Trend toward seizure:** {trend_r} &nbsp;|&nbsp; r={r_r:.2f} &nbsp;|&nbsp; p={p_r:.3f}",
                unsafe_allow_html=True)

    fig_rate = go.Figure()
    fig_rate.add_trace(go.Bar(
        x=df_bins["t_rel_min"],
        y=df_bins["burst_count"],
        marker_color="#3B82F6",
        name="Bursts/min",
    ))
    # Trend line
    x_line = np.array([df_bins["t_rel_min"].min(), df_bins["t_rel_min"].max()])
    y_line = slope_r * x_line + _
    fig_rate.add_trace(go.Scatter(
        x=x_line, y=y_line, mode="lines",
        line=dict(color="red", dash="dash", width=2), name="Trend",
    ))
    fig_rate.add_vline(x=0, line_color="red", line_dash="dot", annotation_text="Seizure")
    fig_rate.update_layout(
        xaxis_title="Minutes relative to seizure onset",
        yaxis_title="Burst count per minute",
        height=300, margin=dict(t=20, b=20), showlegend=False,
    )
    st.plotly_chart(fig_rate, use_container_width=True)

    # ── Plot 2: Burst Duration ──
    st.markdown("#### 2 · Burst Duration (seconds)")
    slope_d, intercept_d, r_d, p_d, _ = linregress(df_iid["t_rel_min"], df_iid["duration_ms"] / 1000)
    trend_d = ("↑ Lengthening" if slope_d < 0 and p_d < 0.1 else
               "↓ Shortening"  if slope_d > 0 and p_d < 0.1 else "No clear trend")
    st.markdown(f"**Trend toward seizure:** {trend_d} &nbsp;|&nbsp; r={r_d:.2f} &nbsp;|&nbsp; p={p_d:.3f}",
                unsafe_allow_html=True)

    fig_dur = go.Figure()
    fig_dur.add_trace(go.Scatter(
        x=df_iid["t_rel_min"],
        y=df_iid["duration_ms"] / 1000,
        mode="markers",
        marker=dict(color="#10B981", size=6, opacity=0.6),
        name="Burst duration",
    ))
    x_line2 = np.array([df_iid["t_rel_min"].min(), df_iid["t_rel_min"].max()])
    fig_dur.add_trace(go.Scatter(
        x=x_line2, y=slope_d * x_line2 + intercept_d, mode="lines",
        line=dict(color="red", dash="dash", width=2), name="Trend",
    ))
    fig_dur.add_vline(x=0, line_color="red", line_dash="dot", annotation_text="Seizure")
    fig_dur.update_layout(
        xaxis_title="Minutes relative to seizure onset",
        yaxis_title="Burst duration (s)",
        height=300, margin=dict(t=20, b=20), showlegend=False,
    )
    st.plotly_chart(fig_dur, use_container_width=True)

    # ── Plot 3: E/I Ratio ──
    if has_ei:
        st.markdown("#### 3 · E/I Ratio (spike power / slow-wave power)")
        st.caption("גבוה = דומיננטיות עירור | נמוך = דומיננטיות עיכוב")

        slope_ei, intercept_ei, r_ei, p_ei, _ = linregress(df_iid["t_rel_min"], df_iid["ei_ratio"])
        trend_ei = ("↑ More excitation" if slope_ei < 0 and p_ei < 0.1 else
                    "↓ More inhibition" if slope_ei > 0 and p_ei < 0.1 else "No clear trend")
        st.markdown(f"**Trend toward seizure:** {trend_ei} &nbsp;|&nbsp; r={r_ei:.2f} &nbsp;|&nbsp; p={p_ei:.3f}",
                    unsafe_allow_html=True)

        fig_ei = go.Figure()
        fig_ei.add_trace(go.Scatter(
            x=df_iid["t_rel_min"],
            y=df_iid["ei_ratio"],
            mode="markers",
            marker=dict(
                color=df_iid["ei_ratio"],
                colorscale="RdBu_r",
                size=7,
                opacity=0.7,
                colorbar=dict(title="E/I ratio"),
                showscale=True,
            ),
            name="E/I ratio",
        ))
        x_line3 = np.array([df_iid["t_rel_min"].min(), df_iid["t_rel_min"].max()])
        fig_ei.add_trace(go.Scatter(
            x=x_line3, y=slope_ei * x_line3 + intercept_ei, mode="lines",
            line=dict(color="black", dash="dash", width=2), name="Trend",
        ))
        fig_ei.add_vline(x=0, line_color="red", line_dash="dot", annotation_text="Seizure")
        fig_ei.add_hline(y=1.0, line_color="gray", line_dash="dot",
                         annotation_text="E=I", annotation_position="right")
        fig_ei.update_layout(
            xaxis_title="Minutes relative to seizure onset",
            yaxis_title="E/I ratio",
            height=320, margin=dict(t=20, b=20), showlegend=False,
        )
        st.plotly_chart(fig_ei, use_container_width=True)

        # ── Per-burst E/I boxplot by 3-minute windows ──
        st.markdown("#### E/I Ratio Distribution by Time Window")
        df_iid["window"] = df_iid["t_rel_min"].apply(
            lambda t: f"{int(np.floor(t/3))*3} to {int(np.floor(t/3))*3+3} min"
        )
        windows_sorted = sorted(df_iid["window"].unique(),
                                key=lambda w: int(w.split(" ")[0]))
        fig_box_ei = go.Figure()
        for w in windows_sorted:
            vals = df_iid.loc[df_iid["window"]==w, "ei_ratio"].values
            fig_box_ei.add_trace(go.Box(
                y=vals, name=w, boxmean=True,
                marker_color="#6366F1",
            ))
        fig_box_ei.update_layout(
            xaxis_title="Time window (min before seizure)",
            yaxis_title="E/I ratio",
            showlegend=False, height=320, margin=dict(t=20, b=20),
        )
        st.plotly_chart(fig_box_ei, use_container_width=True)

    # ── Spectral profile across all bursts ──
    spectral_cols = [c for c in ["delta_rel","theta_rel","alpha_rel","beta_rel","gamma_rel"]
                     if c in df_iid.columns]
    if spectral_cols:
        st.markdown("#### 4 · Mean Spectral Profile of Bursts")
        st.caption("Relative power per band (Welch PSD, normalized — no 1/f bias)")

        band_labels = {"delta_rel":"δ (1-4Hz)","theta_rel":"θ (4-8Hz)",
                       "alpha_rel":"α (8-13Hz)","beta_rel":"β (13-30Hz)","gamma_rel":"γ (30-70Hz)"}
        means = {band_labels[c]: df_iid[c].mean() for c in spectral_cols}

        fig_spec = go.Figure(go.Bar(
            x=list(means.keys()),
            y=list(means.values()),
            marker_color=["#6366F1","#3B82F6","#10B981","#F59E0B","#EF4444"],
            text=[f"{v:.1%}" for v in means.values()],
            textposition="outside",
        ))
        fig_spec.update_layout(
            yaxis_title="Relative power",
            height=300, margin=dict(t=20,b=20),
        )
        st.plotly_chart(fig_spec, use_container_width=True)

        # Spectral evolution toward seizure
        st.markdown("#### 5 · Spectral Evolution Toward Seizure")
        fig_spec_ev = go.Figure()
        colors_sp = ["#6366F1","#3B82F6","#10B981","#F59E0B","#EF4444"]
        for ci, (col, label) in enumerate(band_labels.items()):
            if col not in df_iid.columns: continue
            y_smooth = df_iid[col].rolling(3, min_periods=1).mean()
            fig_spec_ev.add_trace(go.Scatter(
                x=df_iid["t_rel_min"], y=y_smooth,
                mode="lines+markers", name=label,
                line=dict(color=colors_sp[ci], width=2),
                marker=dict(size=5),
            ))
        fig_spec_ev.add_vline(x=0, line_color="red", line_dash="dot",
                              annotation_text="Seizure")
        fig_spec_ev.update_layout(
            xaxis_title="Minutes relative to seizure",
            yaxis_title="Relative power (3-burst rolling mean)",
            height=350, margin=dict(t=20,b=20),
        )
        st.plotly_chart(fig_spec_ev, use_container_width=True)

    # ── Summary table ──
    st.markdown("#### Burst Summary Table")
    display_cols = ["t_rel_min", "duration_ms", "peak_z"]
    if has_ei:
        display_cols += ["ei_ratio", "spike_power", "slow_power"]
    spec_disp = [c for c in spectral_cols if c in df_iid.columns] if spectral_cols else []
    display_cols += spec_disp
    df_display = df_iid[display_cols].copy().round(3)
    rename = {"t_rel_min":"Min before sz","duration_ms":"Duration (ms)","peak_z":"GFP Z",
               "ei_ratio":"E/I ratio","spike_power":"Spike rel.power","slow_power":"SW rel.power",
               "delta_rel":"δ","theta_rel":"θ","alpha_rel":"α","beta_rel":"β","gamma_rel":"γ"}
    df_display = df_display.rename(columns=rename)
    st.dataframe(df_display, use_container_width=True, height=300)


# ── Annotation inspector ──────────────────────────────────────────────────────

with st.expander("📋 All EDF Annotations"):
    ann_rows = [{"onset_s": a["onset"], "duration_s": a["duration"],
                 "description": a["description"]} for a in raw.annotations]
    st.dataframe(pd.DataFrame(ann_rows), use_container_width=True)


# ── Export ────────────────────────────────────────────────────────────────────

st.divider()
st.subheader("💾 Export Results")

col_dl1, col_dl2 = st.columns(2)
with col_dl1:
    csv_iid = df_iid.to_csv(index=False).encode()
    st.download_button("Download IID table (CSV)", csv_iid, "iid_detections.csv", "text/csv")
with col_dl2:
    if not df_psd.empty:
        csv_psd = df_psd.to_csv(index=False).encode()
        st.download_button("Download spectral table (CSV)", csv_psd, "iid_spectral.csv", "text/csv")

st.caption("IID Analyzer v1.0 | Rambam Epilepsy Unit | Herskovitz Lab")


# ── IID Visual Validation ─────────────────────────────────────────────────────

st.divider()
st.subheader("🔍 IID Visual Validation")
st.caption("בדוק כל התפרצות ויזואלית — גלול בין ה-IIDs וסמן כנכון/שגוי")

if len(df_iid) == 0:
    st.info("No IIDs to display.")
else:
    # Controls
    col_v1, col_v2, col_v3 = st.columns([2, 2, 2])
    with col_v1:
        context_ms_viz = st.slider("Context around IID (ms)", 100, 1000, 300, 50,
                                    key="viz_context")
    with col_v2:
        n_per_page = st.selectbox("IIDs per page", [5, 10, 20], index=1, key="viz_n")
    with col_v3:
        viz_channels = st.multiselect(
            "Channels to display",
            options=selected_chs,
            default=selected_chs[:min(6, len(selected_chs))],
            key="viz_chs",
        )

    if not viz_channels:
        st.warning("בחר לפחות ערוץ אחד לתצוגה.")
    else:
        # Pagination
        total_iids = len(df_iid)
        n_pages = max(1, int(np.ceil(total_iids / n_per_page)))
        page = st.number_input("עמוד", min_value=1, max_value=n_pages, value=1, step=1,
                                key="viz_page")
        st.caption(f"מציג IIDs {(page-1)*n_per_page+1}–{min(page*n_per_page, total_iids)} מתוך {total_iids}")

        # Sort IIDs by time for display
        iids_sorted = sorted(iids_raw, key=lambda x: x["onset_sec"])
        page_iids = iids_sorted[(page-1)*n_per_page : page*n_per_page]

        ctx_samps = int(context_ms_viz / 1000 * sfreq)
        viz_ch_indices = [raw.ch_names.index(c) for c in viz_channels]

        # Scale for display (µV)
        scale = 1e6

        for i, iid in enumerate(page_iids):
            global_idx = (page-1)*n_per_page + i
            iid_abs_sec = iid["onset_sec"]

            # Sample indices in full recording
            pk_samp  = int(iid_abs_sec * sfreq)
            s_start  = max(0, pk_samp - ctx_samps)
            s_end    = min(raw._data.shape[1], pk_samp + ctx_samps)

            seg = raw._data[viz_ch_indices, s_start:s_end] * scale
            t_axis = (np.arange(seg.shape[1]) - (pk_samp - s_start)) / sfreq * 1000  # ms

            # Offset channels for butterfly display
            spacing = np.percentile(np.abs(seg), 95) * 3 + 1
            if spacing < 10:
                spacing = 50.0

            fig_v = go.Figure()
            colors = px.colors.qualitative.Plotly

            for ci, ch_name in enumerate(viz_channels):
                offset = ci * spacing
                fig_v.add_trace(go.Scatter(
                    x=t_axis,
                    y=seg[ci] + offset,
                    mode="lines",
                    name=ch_name,
                    line=dict(color=colors[ci % len(colors)], width=1),
                    hovertemplate=f"<b>{ch_name}</b><br>%{{x:.0f}} ms<br>%{{y:.1f}} µV<extra></extra>",
                ))

            # Mark IID onset
            fig_v.add_vline(x=0, line_color="red", line_width=2,
                            annotation_text="IID", annotation_position="top")
            # Mark discharge end
            dur_ms = iid["duration_ms"]
            fig_v.add_vline(x=dur_ms, line_color="orange", line_dash="dash",
                            line_width=1)
            # Shade discharge window
            fig_v.add_vrect(x0=0, x1=dur_ms,
                            fillcolor="red", opacity=0.07, line_width=0)

            t_rel_min = (iid_abs_sec - sz_onset) / 60
            fig_v.update_layout(
                title=dict(
                    text=f"IID #{global_idx+1} | {iid_abs_sec:.1f}s ({t_rel_min:.1f} min before seizure) | "
                         f"Duration: {dur_ms:.0f}ms | Z-score: {iid['peak_z']:.1f}",
                    font=dict(size=12),
                ),
                height=200 + len(viz_channels) * 25,
                margin=dict(t=40, b=20, l=60, r=20),
                showlegend=True,
                legend=dict(orientation="h", y=-0.25, font=dict(size=10)),
                xaxis=dict(title="ms relative to IID onset", zeroline=True,
                           zerolinecolor="red", zerolinewidth=1),
                yaxis=dict(showticklabels=False, title="Channels (offset)"),
                plot_bgcolor="white",
                paper_bgcolor="white",
            )
            fig_v.update_xaxes(gridcolor="#f0f0f0")

            st.plotly_chart(fig_v, use_container_width=True, key=f"iid_viz_{global_idx}")

            # Quick accept/reject buttons (stored in session state)
            if f"iid_label_{global_idx}" not in st.session_state:
                st.session_state[f"iid_label_{global_idx}"] = "unreviewed"

            col_a, col_r, col_u, col_status = st.columns([1, 1, 1, 4])
            with col_a:
                if st.button("✅ אמיתי", key=f"accept_{global_idx}"):
                    st.session_state[f"iid_label_{global_idx}"] = "accepted"
            with col_r:
                if st.button("❌ artifact", key=f"reject_{global_idx}"):
                    st.session_state[f"iid_label_{global_idx}"] = "rejected"
            with col_u:
                if st.button("❓ לא ברור", key=f"unsure_{global_idx}"):
                    st.session_state[f"iid_label_{global_idx}"] = "unsure"
            with col_status:
                label = st.session_state[f"iid_label_{global_idx}"]
                color = {"accepted": "🟢", "rejected": "🔴", "unsure": "🟡", "unreviewed": "⚪"}
                st.markdown(f"{color[label]} **{label}**")

            st.divider()

        # Summary of reviewed
        all_labels = {i: st.session_state.get(f"iid_label_{i}", "unreviewed")
                      for i in range(total_iids)}
        n_accepted  = sum(1 for v in all_labels.values() if v == "accepted")
        n_rejected  = sum(1 for v in all_labels.values() if v == "rejected")
        n_unsure    = sum(1 for v in all_labels.values() if v == "unsure")
        n_reviewed  = n_accepted + n_rejected + n_unsure

        st.markdown(f"**סטטוס ביקורת:** ✅ {n_accepted} אמיתיים | ❌ {n_rejected} artifacts | "
                    f"❓ {n_unsure} לא ברור | ⚪ {total_iids - n_reviewed} לא נבדקו")

        if n_reviewed > 0:
            precision = n_accepted / (n_accepted + n_rejected) * 100 if (n_accepted + n_rejected) > 0 else 0
            st.metric("Precision (מתוך מה שנבדק)", f"{precision:.0f}%")

            # Export with labels
            labels_list = [all_labels.get(i, "unreviewed") for i in range(total_iids)]
            df_iid_labeled = df_iid.copy()
            df_iid_labeled["validation_label"] = labels_list
            csv_labeled = df_iid_labeled.to_csv(index=False).encode()
            st.download_button(
                "💾 הורד IID table עם תוויות ביקורת",
                csv_labeled,
                "iid_validated.csv",
                "text/csv",
                key="dl_validated",
            )
