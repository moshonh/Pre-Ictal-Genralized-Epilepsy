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
    Threshold-based IID detector on average-referenced generalized signal.

    Strategy:
    1. Compute global mean across channels (generalised epilepsy → synchronous)
    2. Z-score the absolute signal
    3. Peak-finding with refractory period
    4. For each peak, extract discharge window and measure duration
    """
    # Average reference across channels
    avg = np.mean(data, axis=0)

    # Absolute value envelope
    abs_sig = np.abs(avg)

    # Z-score (robust)
    med = np.median(abs_sig)
    mad = np.median(np.abs(abs_sig - med)) + 1e-12
    z = (abs_sig - med) / mad

    # Peak detection
    min_distance = int(refractory_ms / 1000 * sfreq)
    peaks, props = signal.find_peaks(z, height=z_thresh, distance=min_distance)

    iids = []
    min_samps = int(min_dur_ms / 1000 * sfreq)
    max_samps = int(max_dur_ms / 1000 * sfreq)

    for pk in peaks:
        # Find extent of discharge above half-max threshold
        half = z[pk] * 0.3
        left = pk
        while left > 0 and z[left] > half:
            left -= 1
        right = pk
        while right < len(z) - 1 and z[right] > half:
            right += 1

        dur_samps = right - left
        if min_samps <= dur_samps <= max_samps:
            iids.append({
                "onset_sample": left,
                "peak_sample": pk,
                "offset_sample": right,
                "onset_sec": left / sfreq,
                "duration_ms": dur_samps / sfreq * 1000,
                "peak_z": float(z[pk]),
                "peak_amp": float(abs_sig[pk]),
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
