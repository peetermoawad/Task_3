"""
Uber Fare Prediction — Flask App (Task 3)
Loads the trained Pipeline from Task 2/3 and serves predictions via a simple web UI.
"""

import os
import math
import joblib
import numpy as np
import pandas as pd
from datetime import datetime
from flask import Flask, render_template, request, flash

app = Flask(__name__)
app.secret_key = "uber-fare-secret-key"

# ---------------------------------------------------------------------------
# Load model + helpers once at startup
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(BASE_DIR, "uber_fare_model.pkl")
FEATURE_PATH = os.path.join(BASE_DIR, "feature_columns.pkl")
BINS_PATH = os.path.join(BASE_DIR, "segment_bins.pkl")

model = None
feature_columns = None
segment_bins = None

def load_artifacts():
    global model, feature_columns, segment_bins
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Model file not found: {MODEL_PATH}\n"
            "Run Task3_Model_Comparison.ipynb first to generate it."
        )
    model = joblib.load(MODEL_PATH)
    feature_columns = joblib.load(FEATURE_PATH) if os.path.exists(FEATURE_PATH) else None
    segment_bins = joblib.load(BINS_PATH) if os.path.exists(BINS_PATH) else {
        "q33": 1.5, "q66": 3.5, "max_dist": 50.0
    }
    print("Model and artifacts loaded successfully.")
    print("Expected features:", feature_columns)


# NYC airport reference coordinates (approx)
AIRPORTS = {
    "jfk": (40.6413, -73.7781),
    "ewr": (40.6895, -74.1745),
    "lga": (40.7769, -73.8740),
}


def haversine(lat1, lon1, lat2, lon2):
    """Distance in km between two lat/lon points."""
    R = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2) ** 2
    return 2 * R * math.asin(math.sqrt(a))


def bearing(lat1, lon1, lat2, lon2):
    """Initial bearing in radians from point 1 to point 2."""
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dlam = math.radians(lon2 - lon1)
    x = math.sin(dlam) * math.cos(phi2)
    y = math.cos(phi1) * math.sin(phi2) - math.sin(phi1) * math.cos(phi2) * math.cos(dlam)
    return math.atan2(x, y)


def validate_inputs(form):
    """
    Validate form fields. Returns (errors_list, cleaned_dict).
    errors_list is empty if everything is OK.
    """
    errors = []
    cleaned = {}

    # --- distance ---
    try:
        dist = float(form.get("distance", "").strip())
        if dist <= 0:
            errors.append("Distance must be greater than 0.")
        elif dist > 100:
            errors.append("Distance looks unrealistic (> 100 km). Please check.")
        else:
            cleaned["distance"] = dist
    except (ValueError, TypeError):
        errors.append("Distance is required and must be a number.")

    # --- datetime ---
    dt_str = form.get("pickup_datetime", "").strip()
    if not dt_str:
        errors.append("Pickup date & time is required.")
    else:
        try:
            # HTML datetime-local returns "YYYY-MM-DDTHH:MM"
            dt = datetime.strptime(dt_str, "%Y-%m-%dT%H:%M")
            cleaned["hour"] = dt.hour
            cleaned["day"] = dt.day
            cleaned["month"] = dt.month
            cleaned["weekday"] = dt.weekday()  # 0=Mon … 6=Sun
            cleaned["year"] = dt.year
        except ValueError:
            errors.append("Invalid date/time format.")

    # --- coordinates (optional but recommended) ---
    coord_fields = ["pickup_latitude", "pickup_longitude",
                    "dropoff_latitude", "dropoff_longitude"]
    coords_ok = True
    for f in coord_fields:
        raw = form.get(f, "").strip()
        if raw == "":
            coords_ok = False
            break
        try:
            cleaned[f] = float(raw)
        except ValueError:
            errors.append(f"{f.replace('_', ' ').title()} must be a number.")
            coords_ok = False

    if coords_ok:
        # Basic NYC-ish range check
        plat, plon = cleaned["pickup_latitude"], cleaned["pickup_longitude"]
        dlat, dlon = cleaned["dropoff_latitude"], cleaned["dropoff_longitude"]
        if not (40.0 <= plat <= 41.5 and -75.0 <= plon <= -72.0):
            errors.append("Pickup coordinates look outside the NYC area.")
        if not (40.0 <= dlat <= 41.5 and -75.0 <= dlon <= -72.0):
            errors.append("Dropoff coordinates look outside the NYC area.")

    cleaned["_coords_ok"] = coords_ok
    return errors, cleaned


def engineer_features(cleaned):
    """
    Apply the same feature engineering used in Task 2.
    Returns a single-row DataFrame with columns matching the training features.
    """
    dist = cleaned["distance"]
    hour = cleaned["hour"]
    day = cleaned.get("day", 1)
    month = cleaned["month"]
    weekday = cleaned["weekday"]
    year = cleaned["year"]

    is_night = int(hour in list(range(0, 6)) + [22, 23])
    is_weekend = int(weekday in [5, 6])

    # trip_segment from saved quantiles
    q33 = segment_bins["q33"]
    q66 = segment_bins["q66"]
    max_d = segment_bins["max_dist"]
    if dist <= q33:
        trip_segment = 0
    elif dist <= q66:
        trip_segment = 1
    else:
        trip_segment = 2

    # Airport distances & near flags
    if cleaned.get("_coords_ok"):
        plat = cleaned["pickup_latitude"]
        plon = cleaned["pickup_longitude"]
        dlat = cleaned["dropoff_latitude"]
        dlon = cleaned["dropoff_longitude"]

        jfk_dist = haversine(plat, plon, *AIRPORTS["jfk"])
        ewr_dist = haversine(plat, plon, *AIRPORTS["ewr"])
        lga_dist = haversine(plat, plon, *AIRPORTS["lga"])
        # also compute dropoff-side if useful; keep simple: use pickup-side
        brng = bearing(plat, plon, dlat, dlon)

        # near flags: use rough 20th-percentile style threshold (~ few km)
        near_jfk = int(jfk_dist <= 5.0)
        near_ewr = int(ewr_dist <= 5.0)
        near_lga = int(lga_dist <= 5.0)
    else:
        # fallback defaults when user does not provide coordinates
        jfk_dist = ewr_dist = lga_dist = 20.0
        brng = 0.0
        near_jfk = near_ewr = near_lga = 0

    # Other distance-to-landmarks placeholders (if model expects them)
    sol_dist = 15.0
    nyc_dist = 10.0

    row = {
        "hour": hour,
        "day": day,
        "month": month,
        "weekday": weekday,
        "year": year,
        "jfk_dist": jfk_dist,
        "ewr_dist": ewr_dist,
        "lga_dist": lga_dist,
        "sol_dist": sol_dist,
        "nyc_dist": nyc_dist,
        "distance": dist,
        "bearing": brng,
        "is_night": is_night,
        "is_weekend": is_weekend,
        "near_jfk": near_jfk,
        "near_ewr": near_ewr,
        "near_lga": near_lga,
        "trip_segment": trip_segment,
    }

    # Build DataFrame with only the columns the model was trained on
    if feature_columns is not None:
        data = {col: [row.get(col, 0)] for col in feature_columns}
    else:
        data = {k: [v] for k, v in row.items()}

    return pd.DataFrame(data)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    errors, cleaned = validate_inputs(request.form)

    if errors:
        return render_template("index.html", errors=errors, form_data=request.form)

    try:
        X = engineer_features(cleaned)
        pred = model.predict(X)[0]
        # Clamp to a sensible non-negative range
        pred = max(0.0, float(pred))
        prediction = round(pred, 2)

        # Add engineered flags for the results page
        cleaned["is_night"] = int(cleaned["hour"] in list(range(0, 6)) + [22, 23])
        cleaned["is_weekend"] = int(cleaned["weekday"] in [5, 6])
    except Exception as e:
        return render_template(
            "index.html",
            errors=[f"Prediction failed: {str(e)}"],
            form_data=request.form,
        )

    return render_template(
        "result.html",
        prediction=prediction,
        form_data=cleaned,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    load_artifacts()
    print("Starting Flask app on http://127.0.0.1:5000")
    app.run(debug=True, host="0.0.0.0", port=5000)
