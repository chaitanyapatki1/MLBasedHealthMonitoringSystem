
import joblib
import sklearn

model = joblib.load("respiratory_model.pkl")
features = joblib.load("feature_order.pkl")
encodings = joblib.load("feature_encodings.pkl")

print("\nScikit-learn version:", sklearn.__version__)

print("\nMODEL INFORMATION")
print("Model:", model)
print("Expected inputs:", model.n_features_in_)

print("\nFEATURE ORDER")
for i, feature in enumerate(features, start=1):
    print(f"{i}. {feature}")

print("\nFEATURE ENCODINGS")
for feature, mapping in encodings.items():
    print(f"{feature}: {mapping}")

assert len(features) == model.n_features_in_

print("\nVerification successful!")
