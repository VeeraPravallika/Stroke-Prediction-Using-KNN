import sys
import argparse
import warnings
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.exceptions import DataConversionWarning

# -----------------------------
# Load and preprocess dataset
# -----------------------------
data = pd.read_csv("healthcare-dataset-stroke-data.csv")

# Restrict noisy warnings that are already handled by preprocessing
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DataConversionWarning)
warnings.filterwarnings("ignore", message="X does not have valid feature names")

# drop id if present
if "id" in data.columns:
    data = data.drop("id", axis=1)

# handle BMI safely
data["bmi"] = data["bmi"].fillna(data["bmi"].mean())

# safe categorical encoding
for col in ["gender", "ever_married", "work_type", "Residence_type", "smoking_status"]:
    if col in data.columns:
        data[col] = data[col].fillna("Unknown")
        data[col] = LabelEncoder().fit_transform(data[col])

# -----------------------------
# Limit to 1000 rows (balanced)
# -----------------------------
stroke_cases = data[data["stroke"] == 1]
if len(stroke_cases) >= 1000:
    data_1000 = stroke_cases.sample(n=1000, random_state=42)
else:
    needed = 1000 - len(stroke_cases)
    non_stroke_cases = data[data["stroke"] == 0].sample(n=needed, random_state=42)
    data_1000 = pd.concat([stroke_cases, non_stroke_cases]).sample(frac=1, random_state=42)

X = data_1000.drop("stroke", axis=1)
y = data_1000["stroke"]

# -----------------------------
# Function to train & evaluate
# -----------------------------
def train_evaluate(test_ratio):
    print(f"\n----- Train-Test Split: {int((1-test_ratio)*100)}:{int(test_ratio*100)} -----")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_ratio, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    knn = KNeighborsClassifier(n_neighbors=7, weights="distance")
    knn.fit(X_train, y_train)

    y_pred = knn.predict(X_test)

    print("Accuracy :", accuracy_score(y_test, y_pred))
    print("Precision:", precision_score(y_test, y_pred, zero_division=0))
    print("Recall   :", recall_score(y_test, y_pred, zero_division=0))
    print("F1 Score :", f1_score(y_test, y_pred, zero_division=0))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

    return knn, scaler

# -----------------------------
# Evaluate for both ratios
# -----------------------------
model_80, scaler_80 = train_evaluate(0.20)
model_70, scaler_70 = train_evaluate(0.30)

# -----------------------------
# 🔹 INTERACTIVE REAL-TIME PREDICTION
# -----------------------------
print("\n--- Stroke Risk Prediction (Real-Time Input) ---")
parser = argparse.ArgumentParser(add_help=False)
parser.add_argument("--no-interactive", action="store_true", help="Disable interactive prompts")
parser.add_argument("--demo", action="store_true", help="Run a demo prediction using median values")
args, _ = parser.parse_known_args()

interactive = not args.no_interactive
demo = args.demo

if interactive:
    # If stdin is not a TTY, still attempt to prompt but handle EOF gracefully
    if not sys.stdin.isatty():
        print("Note: stdin is not a TTY. Prompting may fail if run non-interactively.")

    print("\n--- Stroke Risk Prediction (Real-Time Input) ---")

    user_input = []
    for col in X.columns:
        while True:
            try:
                raw = input(f"Enter {col}: ")
                value = float(raw)
                user_input.append(value)
                break
            except ValueError:
                print("Please enter a numeric value.")
            except EOFError:
                # Non-interactive stdin; fall back to demo
                print("\nEOF reached during input — falling back to demo prediction.")
                demo = True
                break

    if not demo and len(user_input) == len(X.columns):
        user_input_df = pd.DataFrame([user_input], columns=X.columns)
        user_input_scaled = scaler_80.transform(user_input_df)
        prediction = model_80.predict(user_input_scaled)

        if prediction[0] == 1:
            print("⚠️ High Stroke Risk Detected")
        else:
            print("✅ Low Stroke Risk")

if demo:
    # Build a sample input using medians from the dataset
    sample = X.median()
    print("\n--- Demo prediction using median feature values ---")
    print(sample.to_frame(name="value").T)
    sample_df = pd.DataFrame([sample.values], columns=X.columns)
    sample_scaled = scaler_80.transform(sample_df)
    pred = model_80.predict(sample_scaled)
    print("Prediction:", "High Risk" if pred[0] == 1 else "Low Risk")
