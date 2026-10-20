# MadhuTwin — Person 2 handoff (JSON model)

## What changed

- The adapter is `src/serving/predict_api.py`, imported by the existing backend when `MADHUTWIN_MODEL=real`.
- It loads `models/xgb_all_cgmacros.json` using `xgboost.XGBClassifier().load_model(...)`. It does not use `joblib` or need a `.joblib` file.
- `models/xgb_all_cgmacros_metadata.json` is the supplied metadata.
- `src/serving/mock_predict.py` is included unchanged as a reference. Keep the main project's existing `src/serving/mock_predict.py`, `src/serving/mock_data.py`, `src/serving/timeseries.py`, `src/contracts.py`, and `src/serving/backend.py` unchanged unless independently reviewed.
- `reference/backend.py` and `reference/contracts.py` are comparison-only. Do not overwrite the live project files just because these copies are in the ZIP.

## Model capability and limitations

The supplied metadata states: dataset `CGMacros`; target `y_future_max >= threshold`; threshold 180 mg/dL; 27,833 samples; 514 features; spike prevalence about 7.08%. The JSON loads as a binary classifier with a 514-feature input (72 steps × 7 sequence channels + 10 static features).

The adapter assembles 72×7 historical channels, flattens them in row-major order, then appends 10 static features in metadata order. It uses a 5-minute grid, sums meal macros by bin, and forward-fills glucose/HR/MET only for up to 30 minutes. It refuses to silently fill longer missing history. The 514-feature count and metadata support this feature shape, but the supplied metadata does not itself document all preprocessing details. Verify preprocessing against the original training/export code if that code becomes available.

**Label-boundary caveat:** the supplied metadata describes the target as `y_future_max >= 180`, while the available `build_datasets.py` uses `np.max(fut) > 180` when creating labels. I cannot confirm from the model JSON alone which exact label-generation revision produced this artifact. Check the training NPZ/export if exact reproducibility for values equal to 180 matters.

The model predicts **future glucose-spike probability only**. It does not produce glucose values at 30/60/120 minutes, quantile bands, hypoglycaemia probability, SHAP values, or attention weights. To preserve the existing `Forecast` contract, the adapter uses XGBoost for `spike_prob` and calls the unchanged mock predictor for the other mandatory forecast fields. Overall `Forecast.provenance` therefore remains `MOCK`, and `model_version` identifies the XGBoost + mock combination. Do not describe this as an all-real digital twin or clinically validated prediction.

No measured evaluation reports were supplied. Do not copy placeholder/mock reports into the real reports directory. The 0.5 probability cutoff is a technical default only, not selected on validation data and not clinically validated.

## Files to merge into the main repository

Copy `src/serving/predict_api.py`, `models/xgb_all_cgmacros.json`, and `models/xgb_all_cgmacros_metadata.json` into those relative locations in MadhuTwin. Keep the existing project infrastructure, especially `src/contracts.py`, `src/serving/backend.py`, `mock_predict.py`, `mock_data.py`, and `timeseries.py`. The `reference/` copies are for comparison only.

## Checks to run (from the full repository root)

1. Install the model dependencies in the same Python environment as MadhuTwin:

   ```bash
   python -m pip install -r requirements-model.txt
   ```

   This is an additional dependency file, not a replacement for the project's existing requirements.

2. Test the JSON model by itself:

   ```bash
   python -c "import xgboost as xgb, numpy as np; m=xgb.XGBClassifier(); m.load_model('models/xgb_all_cgmacros.json'); print('features:', m.n_features_in_); print('smoke probability:', float(m.predict_proba(np.zeros((1,514),dtype=np.float32))[0,1]))"
   ```

   Expected: `features: 514` and a finite probability between 0 and 1. The all-zero vector is a loading smoke test only, not a meaningful patient prediction.

3. Run artifact and feature-construction tests:

   ```bash
   python -m pytest tests/test_model_artifact.py tests/test_predict_api_features.py -q
   ```

   Run after merging into the complete repository because the feature test imports `src.contracts`.

4. Run the repository's existing tests, then start with the classifier selected.

   **Windows PowerShell**
   ```powershell
   $env:MADHUTWIN_BACKEND = "mock"
   $env:MADHUTWIN_MODEL = "real"
   streamlit run dashboard/app.py
   ```

   **macOS/Linux**
   ```bash
   MADHUTWIN_BACKEND=mock MADHUTWIN_MODEL=real streamlit run dashboard/app.py
   ```

   This intentionally uses mock data and the real XGBoost spike classifier. The remainder of the forecast stays mock-generated and must remain disclosed as such.

5. To return to the original demo, set `MADHUTWIN_MODEL=mock` or remove that variable, then restart Streamlit.

## If a check fails

- `model artifact not found`: ensure JSON is at `<repository-root>/models/xgb_all_cgmacros.json`.
- XGBoost error: ensure `python` and `pip` refer to the same environment and use the version in `requirements-model.txt`.
- Missing/unresolved history: model needs six hours (72 five-minute steps) of valid past data. Contract validation guarantees only four hours, so some real data may be insufficient.
- Tests only verify deserialization and input shape. They do not prove predictive performance. Probabilities should not be used in a clinical decision.

Research PoC only. Synthetic/open data only. Not a medical device. Not medical advice.
