import streamlit as st

st.set_page_config(page_title="PPG Annotation Tool", layout="wide")

# --- Header ---
st.title("PPG Signal Annotation Tool")
st.write("A tool for visualizing and annotating PPG signal quality.")

st.divider()

# --- Layout ---
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Signal Viewer")

    # Placeholder instead of real data
    st.info("Signal plot will appear here")

    st.line_chart([0, 1, 0.5, 0.8, 0.3])  # fake placeholder signal

with col2:
    st.subheader("Annotation Panel")

    st.write("Select signal quality:")

    label = st.radio(
        "Label",
        ["Good", "Usable", "Bad"]
    )

    st.button("Save Label")

st.divider()

# --- Feature Panel ---
st.subheader("Extracted Features (placeholder)")

st.json({
    "SNR": "TBD",
    "IBI stability": "TBD",
    "Skewness": "TBD",
    "Template correlation": "TBD"
})