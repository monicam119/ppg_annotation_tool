# PPG Annotation Tool

A Streamlit-based tool for preprocessing, feature extraction, visualization, and annotation of photoplethysmography (PPG) signals to support signal quality assessment and supervised machine learning workflows.

## Features

- PPG preprocessing and filtering
- Signal segmentation into fixed-length windows
- Feature extraction for signal quality assessment
- Interactive visualization of raw and filtered signals
- Foundation for ML-assisted signal quality annotation

## Current Signal Quality Features

- Interbeat interval (IBI) stability
- Signal-to-noise ratio (SNR)
- Skewness
- Template correlation
- Perfusion index (PI)

## Tech Stack

- Python
- Streamlit
- NumPy
- SciPy
- Scikit-learn
- Plotly
- WFDB

## Installation

Clone the repository:

```bash
git clone https://github.com/monicam119/ppg-annotation-tool.git
cd ppg-annotation-tool
```

Create a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the Streamlit app:

```bash
streamlit run app.py
```

## Project Structure

```text
ppg-annotation-tool/
│
├── app.py
├── requirements.txt
├── README.md
└── scripts/
  ├── preprocessing.py
  └── features.py
```

## Future Work

- Multi-class signal quality classification
- ML-assisted annotation workflows
- User-guided model retraining
- Cross-dataset validation and transferability testing

## Notes

Raw physiological datasets (e.g., MIMIC) are not included in this repository due to dataset licensing.
