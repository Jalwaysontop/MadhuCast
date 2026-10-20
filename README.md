# MadhuTwin Person 2 handoff — native JSON model

Read `PERSON1_HANDOFF.md` before copying these files into the main MadhuTwin repository.

This package includes the supplied native XGBoost model `models/xgb_all_cgmacros.json`, its metadata,
an adapter at `src/serving/predict_api.py`, an unchanged copy of the existing mock predictor,
two tests, and reference copies of `backend.py` and `contracts.py` for comparison.

This is a transitional handoff: the XGBoost artifact predicts future glucose-spike risk only.
Non-classifier forecast fields remain supplied by the existing mock predictor and are labeled
MOCK. It is not a fully real glucose-trajectory, uncertainty, or hypoglycaemia model.
