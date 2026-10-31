
import joblib
import pandas as pd

model = joblib.load("respiratory_model.pkl")
features = joblib.load("feature_order.pkl")
encodings = joblib.load("feature_encodings.pkl")


def encode_answer(column, answer):
    """Convert a verified Yes/No answer using saved encodings."""
    if answer not in ("Yes", "No"):
        raise ValueError(f"Unresolved answer for {column}")

    original_value = "True" if answer == "Yes" else "False"
    classes = encodings[column]

    if isinstance(classes, dict):
        return int(classes[original_value])

    return classes.index(original_value)


def prepare_ml_input(sensor, survey, manual):

    cold = encode_answer("Cold", survey["Cold"])
    cough = encode_answer("Cough", survey["Cough"])

    dehydration = encode_answer(
        "Dehydration", survey["Dehydration"]
    )

    overdose = encode_answer(
        "Medicine Overdose",
        survey["Medicine Overdose"]
    )

    acidious = encode_answer(
        "Acidious", survey["Acidious"]
    )

    values = {
        "Dehydration": dehydration,
        "Medicine Overdose": overdose,
        "Acidious": acidious,
        "Cold": cold,
        "Cough": cough,
        "Cold .1": cold,
        "Cough.1": cough,

        "Type": (
            dehydration + overdose + acidious
            + cold + cough
        ),

        "Temperature": (
            manual["body_temperature_c"] * 9 / 5 + 32
        ),

        "Heart Rate": sensor["heart_rate"],
        "Pulse": manual["Pulse"],

        "BPSYS": manual["BPSYS"],
        "BPDIA": manual["BPDIA"],

        "Respiratory Rate": manual["Respiratory Rate"],

        "Oxygen Saturation": sensor["spo2"] / 100,

        # Unknown original measurement definition
        "PH": None
    }

    missing = [
        feature for feature in features
        if values.get(feature) is None
    ]

    if missing:
        return None, missing

    dataframe = pd.DataFrame(
        [[values[f] for f in features]],
        columns=features
    )

    return dataframe, []


def predict(dataframe):
    if dataframe is None:
        raise ValueError("Incomplete model input.")

    prediction = int(model.predict(dataframe)[0])

    labels = {
        0: "Normal",
        1: "Mild",
        2: "Severe",
        3: "Chronic"
    }

    return labels[prediction]
