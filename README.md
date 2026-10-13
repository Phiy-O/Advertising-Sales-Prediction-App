# Advertising Sales Prediction App

An interactive web application built with Streamlit to estimate sales based on advertising spend across TV and Radio channels using a trained linear regression model.

## Features

- Predict sales based on TV and Radio advertising budget inputs
- Uses a pre-trained regression model and feature scaler (`MinMaxScaler`)
- Clean and intuitive Streamlit interface
- Input validation and budget summary display

## Dataset

The model is trained on the Advertising dataset available on Kaggle:
- **Dataset Link**: [Advertising Dataset on Kaggle](https://www.kaggle.com/datasets/ashydv/advertising-dataset)

## Installation & Setup

### Prerequisites

- Python 3.11 to 3.13 (recommended; Python 3.14 may encounter build dependencies issues for scikit-learn wheels)
- Git

### 1. Clone the Repository

```bash
git clone https://github.com/USERNAME/Advertising-Sales-Prediction-App.git
cd Advertising-Sales-Prediction-App
```

### 2. Create and Activate Virtual Environment

**Windows (PowerShell):**
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**macOS/Linux:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Run the Streamlit App

```bash
streamlit run app.py
```

The app will open automatically in your browser at `http://localhost:8501`.

## Project Structure

```text
├── model/
│   ├── model_regresi_advertising.pkl   # Trained regression model
│   └── scaler_advertising.pkl          # Scaler artifact
├── app.py                              # Streamlit application
├── requirements.txt                    # Python dependencies
├── .gitignore                          # Git ignore rules
└── README.md                           # Documentation
```

## Credits

- **Dataset**: Provided by [Ashish](https://www.kaggle.com/ashydv) on Kaggle.
- **Frameworks & Libraries**:
  - [Streamlit](https://streamlit.io/)
  - [scikit-learn](https://scikit-learn.org/)
  - [Joblib](https://joblib.readthedocs.io/)
  - [NumPy](https://numpy.org/)
