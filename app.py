import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy import signal, stats, interpolate
import sys
import os

# ── page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="PPG Signal Annotation Tool",
    layout="wide",
)

# ── custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .block-container { padding-top: 1.5rem; padding-bottom: 2rem; }
    div[data-testid="stSidebar"] { background: #f8f9fa; }
    .metric-card {
        background: #ffffff;
        border: 1px solid #e9ecef;
        border-radius: 10px;
        padding: 0.75rem 1rem;
        margin-bottom: 0.5rem;
    }
    .metric-label { font-size: 11px; color: #6c757d; text-transform: uppercase; letter-spacing: 0.05em; }
    .metric-value { font-size: 20px; font-weight: 600; color: #212529; }
    .badge-good {
        background: #d4edda; color: #155724;
        padding: 3px 10px; border-radius: 20px; font-size: 12px; font-weight: 600;
    }
    .badge-bad {
        background: #f8d7da; color: #721c24;
        padding: 3px 10px; border-radius: 20px; font-size: 12px; font-weight: 600;
    }
    .badge-unlabelled {
        background: #e9ecef; color: #495057;
        padding: 3px 10px; border-radius: 20px; font-size: 12px; font-weight: 600;
    }
    .stButton > button {
        border-radius: 8px;
        font-weight: 500;
        transition: all 0.15s ease;
    }
</style>
""", unsafe_allow_html=True)

# ── session state defaults ─────────────────────────────────────────────────────
def init_state():
    defaults = {
        "windows": None,          # list of 1-D numpy arrays, one per window
        "labels": None,           # list: "good" | "bad" | None
        "current_idx": 0,
        "n_manual": 100,
        "algo_run": False,
        "features": None,         # DataFrame, one row per window
        "show_features": ["SNR", "PI", "Skewness"],
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_state()

# ── helper: compute per-window features ───────────────────────────────────────
def compute_features(windows, fs=125):
    """
    Compute signal quality features for each window.
    Extend this function as your algorithm grows.
    """
    rows = []
    for w in windows:
        w = np.asarray(w, dtype=float)
        # SNR (ratio of signal power to noise power via a simple bandpass proxy)
        b, a = signal.butter(3, [0.4, 8.0], btype="band", fs=fs)
        filtered = signal.filtfilt(b, a, w)
        noise = w - filtered
        snr = 10 * np.log10(np.var(filtered) / (np.var(noise) + 1e-12))
        # Perfusion Index (proxy: AC/DC ratio)
        pi = (np.max(w) - np.min(w)) / (np.mean(np.abs(w)) + 1e-12)
        sk  = float(stats.skew(w))
        ku  = float(stats.kurtosis(w))
        rms = float(np.sqrt(np.mean(w**2)))
        rows.append({"SNR": round(snr, 2), "PI": round(pi, 3),
                     "Skewness": round(sk, 3), "Kurtosis": round(ku, 3),
                     "RMS": round(rms, 4)})
    return pd.DataFrame(rows)

# ── helper: rule-based overall rating ─────────────────────────────────────────
def overall_rating(row, visible_features):
    """
    Simple threshold logic.  Adjust thresholds to match your domain.
    Returns 'good' or 'bad'.
    """
    votes = []
    if "SNR" in visible_features:
        votes.append(row["SNR"] >= 10)
    if "PI" in visible_features:
        votes.append(0.2 <= row["PI"] <= 5.0)
    if "Skewness" in visible_features:
        votes.append(abs(row["Skewness"]) <= 1.5)
    if "Kurtosis" in visible_features:
        votes.append(row["Kurtosis"] <= 10)
    if "RMS" in visible_features:
        votes.append(row["RMS"] > 0)
    if not votes:
        return "unknown"
    return "good" if (sum(votes) / len(votes)) >= 0.6 else "bad"

# ── helper: algorithm labels remaining windows ─────────────────────────────────
def run_algorithm(windows, features, manual_labels):
    """
    Placeholder algorithm — replace with your real model/classifier here.
    Currently applies the same feature-threshold logic as overall_rating().
    """
    labels = list(manual_labels)
    for i, row in features.iterrows():
        if labels[i] is None:
            labels[i] = overall_rating(row, list(features.columns))
    return labels

# ═══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## ⚙️ Settings")
    st.divider()

    # ── Manual label count ─────────────────────────────────────────────────────
    st.markdown("**Manual label count**")
    st.session_state["n_manual"] = st.number_input(
        "How many windows to label manually",
        min_value=100,
        max_value=10000,
        value=st.session_state["n_manual"],
        step=10,
        label_visibility="collapsed",
    )

    st.divider()

    # ── Feature selection ──────────────────────────────────────────────────────
    st.markdown("**Features to display**")
    all_features = ["Skewness", "IBI Stability", "Template Correlation", "PI", "SNR", "Kurtosis", "Power Ratio"]
    selected_features = []
    for feat in all_features:
        checked = feat in st.session_state["show_features"]
        if st.checkbox(feat, value=checked, key=f"cb_{feat}"):
            selected_features.append(feat)
    st.session_state["show_features"] = selected_features

    st.divider()

    # ── Progress summary ───────────────────────────────────────────────────────
    if st.session_state["labels"] is not None:
        labels = st.session_state["labels"]
        n_total    = len(labels)
        n_labelled = sum(1 for l in labels if l is not None)
        n_good     = sum(1 for l in labels if l == "good")
        n_bad      = sum(1 for l in labels if l == "bad")
        n_target   = min(st.session_state["n_manual"], n_total)

        st.markdown("**Progress**")
        st.progress(min(n_labelled / n_target, 1.0))
        st.markdown(f"`{n_labelled}` / `{n_target}` labelled")
        col1, col2 = st.columns(2)
        col1.metric("Good", n_good)
        col2.metric("Bad",  n_bad)

# ═══════════════════════════════════════════════════════════════════════════════
# MAIN AREA
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown("# PPG Signal Annotation Tool")
st.caption("Upload a dataset, label some signals manually, then let the algorithm handle the rest.")

# ── Step 1: Upload ─────────────────────────────────────────────────────────────
st.markdown("### 1 · Upload dataset")

# ╔══════════════════════════════════════════════════════════════════╗
# ║  PLACEHOLDER — swap this block for your real file loader        ║
# ║  Once you know the file format, replace the demo data below     ║
# ║  with your actual parsing logic (e.g. wfdb.rdrecord, pd.read_csv║
# ╚══════════════════════════════════════════════════════════════════╝
uploaded_file = st.file_uploader(
    "Upload your PPG dataset (format TBD — placeholder)",
    type=None,                 # ← set to e.g. ["csv"] or ["dat"] once known
    help="File format not yet determined. Swap this uploader with your real loader.",
)

if uploaded_file is not None and st.session_state["windows"] is None:
    with st.spinner("Preprocessing…"):
        # ── PLACEHOLDER: generate synthetic windows for now ──────────────
        # Replace everything between the dashes with your real parsing +
        # preprocessing call, e.g.:
        #   import sys; sys.path.append("scripts/")
        #   from preprocess_extract import load_and_preprocess
        #   windows = load_and_preprocess(uploaded_file)
        # ─────────────────────────────────────────────────────────────────
        np.random.seed(42)
        fs = 125
        n_windows = 200
        t = np.linspace(0, 2, fs * 2)
        windows = []
        for i in range(n_windows):
            w  = np.sin(2 * np.pi * 1.2 * t + np.random.uniform(0, 2*np.pi))
            w += 0.3 * np.sin(2 * np.pi * 2.4 * t)
            w += np.random.normal(0, 0.1 + 0.3 * (i % 3 == 0), len(t))
            windows.append(w)
        # ─────────────────────────────────────────────────────────────────

        st.session_state["windows"] = windows
        st.session_state["labels"]  = [None] * len(windows)
        st.session_state["features"] = compute_features(windows, fs=fs)
        st.session_state["current_idx"] = 0
        st.session_state["algo_run"] = False
    st.success(f"Loaded {len(windows)} windows — preprocessing complete.")

# ── Steps 2-4 only shown after upload ─────────────────────────────────────────
if st.session_state["windows"] is not None:
    windows  = st.session_state["windows"]
    labels   = st.session_state["labels"]
    features = st.session_state["features"]
    idx      = st.session_state["current_idx"]
    n_total  = len(windows)
    n_target = min(st.session_state["n_manual"], n_total)

    # ── Step 2: Manual labelling ───────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### 2 · Label signals")

    # Navigation row
    nav_col, _, status_col = st.columns([2, 3, 2])
    with nav_col:
        c1, c2, c3 = st.columns(3)
        if c1.button("◀ Prev", use_container_width=True):
            st.session_state["current_idx"] = max(0, idx - 1)
            st.rerun()
        c2.markdown(f"<div style='text-align:center;padding-top:6px;font-weight:600'>{idx+1} / {n_total}</div>", unsafe_allow_html=True)
        if c3.button("Next ▶", use_container_width=True):
            st.session_state["current_idx"] = min(n_total - 1, idx + 1)
            st.rerun()
    with status_col:
        lbl = labels[idx]
        if lbl == "good":
            st.markdown('<span class="badge-good">✓ Good</span>', unsafe_allow_html=True)
        elif lbl == "bad":
            st.markdown('<span class="badge-bad">✗ Bad</span>', unsafe_allow_html=True)
        else:
            st.markdown('<span class="badge-unlabelled">Unlabelled</span>', unsafe_allow_html=True)

    # Signal plot + features + label buttons
    plot_col, label_col = st.columns([3, 1])

    with plot_col:
        w = windows[idx]
        fig, ax = plt.subplots(figsize=(8, 2.8))
        ax.plot(w, color="#1D9E75", linewidth=1.4)
        ax.set_xlabel("Sample", fontsize=10)
        ax.set_ylabel("Amplitude", fontsize=10)
        ax.set_title(f"Window #{idx + 1}", fontsize=11, fontweight="bold")
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(labelsize=9)
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

        # Feature badges below the plot
        if selected_features and not features.empty:
            row = features.iloc[idx]
            rating = overall_rating(row, selected_features)
            badge_html = ""
            for feat in selected_features:
                badge_html += f'<span style="background:#e9ecef;color:#495057;padding:3px 10px;border-radius:20px;font-size:12px;margin-right:6px;">{feat}: {row[feat]}</span>'
            rating_color = "#d4edda" if rating == "good" else "#f8d7da"
            rating_text  = "#155724" if rating == "good" else "#721c24"
            badge_html += f'<span style="background:{rating_color};color:{rating_text};padding:3px 10px;border-radius:20px;font-size:12px;font-weight:600;">Overall: {rating}</span>'
            st.markdown(badge_html, unsafe_allow_html=True)

    with label_col:
        st.markdown("**Label this window**")
        if st.button("Good", use_container_width=True, type="primary"):
            st.session_state["labels"][idx] = "good"
            # Auto-advance to next unlabelled within manual range
            for i in range(idx + 1, n_target):
                if st.session_state["labels"][i] is None:
                    st.session_state["current_idx"] = i
                    break
            st.rerun()
        if st.button("Bad", use_container_width=True):
            st.session_state["labels"][idx] = "bad"
            for i in range(idx + 1, n_target):
                if st.session_state["labels"][i] is None:
                    st.session_state["current_idx"] = i
                    break
            st.rerun()

        st.markdown("---")
        n_done = sum(1 for l in labels[:n_target] if l is not None)
        st.caption(f"{n_done} / {n_target} done")
        st.progress(min(n_done / n_target, 1.0))

    # ── Step 3: Run algorithm ──────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### 3 · Run algorithm on remaining windows")

    n_done = sum(1 for l in labels[:n_target] if l is not None)
    n_remaining = sum(1 for l in labels if l is None)

    if n_done < n_target:
        st.info(f"Complete {n_target - n_done} more manual labels before running the algorithm.")
    else:
        if not st.session_state["algo_run"]:
            st.write(f"You've finished your {n_target} manual labels. {n_remaining} windows will be auto-labelled.")
            if st.button("▶ Run algorithm", type="primary"):
                with st.spinner("Running algorithm…"):
                    st.session_state["labels"] = run_algorithm(
                        windows, features, st.session_state["labels"]
                    )
                    st.session_state["algo_run"] = True
                st.success("Done! All windows labelled.")
                st.rerun()
        else:
            st.success("Algorithm complete — all windows labelled.")

    # ── Step 4: Download ───────────────────────────────────────────────────────
    if st.session_state["algo_run"]:
        st.markdown("---")
        st.markdown("### 4 · Download results")

        # ╔══════════════════════════════════════════════════════════════════╗
        # ║  PLACEHOLDER — swap this for your real output format           ║
        # ║  Once you know the output format, replace the CSV export below  ║
        # ║  with whatever format your data originally came in              ║
        # ╚══════════════════════════════════════════════════════════════════╝
        df_out = features.copy()
        df_out.insert(0, "window_index", range(len(windows)))
        df_out["label"] = st.session_state["labels"]

        csv_bytes = df_out.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Download labelled dataset (CSV placeholder)",
            data=csv_bytes,
            file_name="ppg_labelled.csv",
            mime="text/csv",
            help="Format is a placeholder — swap with your real output format once known.",
        )
        st.caption("Columns: window index, features, label (good/bad)")

else:
    st.info("Upload a dataset above to get started.")