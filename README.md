# Real Estate Investment Advisor

Classifies whether a property is a **Good Investment** and predicts its **price in 5 years**; Streamlit app + MLflow tracking.

## Run
```bash
pip install -r requirements.txt
python -m src.train          # builds data/cleaned_*.csv, trains, logs to MLflow, registers models (~15-20 min on 1 CPU)
python -m src.eda            # 20 EDA questions -> eda_outputs/
streamlit run app/streamlit_app.py
mlflow ui --backend-store-uri sqlite:///mlflow.db   # experiment runs + registered models
```
Pre-trained models, the MLflow database and the cleaned CSV are already included, so you can launch the app directly.

## Layout
- `src/config.py`, `src/features.py` : cleaning, feature engineering, targets
- `src/eda.py`, `src/train.py`, `src/predict.py`
- `app/streamlit_app.py` : Home, Investment Advisor, Property Explorer, Market Insights, Model Performance, About
- `docs/METHODOLOGY.md` : methodology, results, limitations (**read the limitations**)
