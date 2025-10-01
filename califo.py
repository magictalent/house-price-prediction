# house_price_dl_project.py
# Run in a notebook or python environment. Tested with Python 3.9+, TensorFlow 2.12+ recommended.

# Install needed libs if missing:
# pip install numpy pandas scikit-learn tensorflow matplotlib shap streamlit

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
import shap
import joblib

# 1) Load data (California housing — replaceable with your CSV)
def load_california():
    data = fetch_california_housing(as_frame=False)
    df = pd.DataFrame(data.data, columns=data.feature_names)
    df['MedHouseVal'] = data.target
    return df

df = load_california()
print("shape:", df.shape)
df.head()

# 2) Quick EDA / basic cleaning
print(df.describe().T)
# Check for nulls
print("nulls per column:\n", df.isna().sum())

# If using CSV:
# df = pd.read_csv("your_file.csv"); then set X, y accordingly.

# 3) Feature engineering example
# Create simple interaction features (example)
df['RoomsPerHousehold'] = df['AveRooms'] / (df['HouseAge'] + 1e-6)
df['BedroomsPerRoom'] = df['AveBedrms'] / (df['AveRooms'] + 1e-6)

# Optional: log transform target if very skewed
# df['MedHouseVal'] = np.log1p(df['MedHouseVal'])

# 4) Train/test split
X = df.drop(columns=['MedHouseVal'])
y = df['MedHouseVal'].values

X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.25, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

print(X_train.shape, X_val.shape, X_test.shape)

# 5) Preprocessing pipeline
numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
# There are no nominal categorical columns in this dataset; placeholder if you have any:
categorical_features = []  # e.g. ['neighborhood']

numeric_transformer = Pipeline(steps=[
    ('scaler', StandardScaler())
])

preprocessor = ColumnTransformer(transformers=[
    ('num', numeric_transformer, numeric_features),
    # ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
], remainder='drop')

# Fit scaler pipeline
preprocessor.fit(X_train)

# Transform
X_train_proc = preprocessor.transform(X_train)
X_val_proc = preprocessor.transform(X_val)
X_test_proc = preprocessor.transform(X_test)

# 6) Build Keras model (dense)
input_dim = X_train_proc.shape[1]

def build_model(input_dim, dropout_rate=0.2, l2=1e-4):
    inp = layers.Input(shape=(input_dim,))
    x = layers.Dense(256, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(l2))(inp)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(dropout_rate)(x)
    x = layers.Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(l2))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(dropout_rate)(x)
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dense(1, activation='linear')(x)
    model = models.Model(inputs=inp, outputs=x)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
                  loss='mse',
                  metrics=['mae'])
    return model

model = build_model(input_dim)
model.summary()

# 7) Callbacks
checkpoint_path = "best_model.h5"
cb = [
    callbacks.ModelCheckpoint(checkpoint_path, save_best_only=True, monitor='val_loss'),
    callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
    callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6)
]

# 8) Train
history = model.fit(
    X_train_proc, y_train,
    validation_data=(X_val_proc, y_val),
    epochs=100,
    batch_size=64,
    callbacks=cb,
    verbose=2
)

# 9) Evaluation
def evaluate_model(model, X, y, label="test"):
    preds = model.predict(X).squeeze()
    mse = mean_squared_error(y, preds)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y, preds)
    r2 = r2_score(y, preds)
    print(f"{label} RMSE: {rmse:.4f}, MAE: {mae:.4f}, R2: {r2:.4f}")
    return preds

preds_test = evaluate_model(model, X_test_proc, y_test, "test")

# Plot predictions vs true
plt.figure(figsize=(6,6))
plt.scatter(y_test, preds_test, alpha=0.3)
plt.xlabel("True target")
plt.ylabel("Predicted")
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.title("Predicted vs True")
plt.show()

# 10) Save pipeline (preprocessor + model)
os.makedirs("artifacts", exist_ok=True)
joblib.dump(preprocessor, "artifacts/preprocessor.joblib")
model.save("artifacts/house_price_model.keras")   # TF SavedModel format

# 11) Explainability with SHAP (KernelExplainer can be slow; we use DeepExplainer for TF)
# NOTE: For DeepExplainer, background data size should be small. If issues, use Tree/Kernel explainers.
background = X_train_proc[np.random.choice(X_train_proc.shape[0], 200, replace=False)]
explainer = shap.DeepExplainer(model, background)
shap_values = explainer.shap_values(X_test_proc[:200])  # returns list; for single output it's [shap_vals]

# shap summary plot
shap.summary_plot(shap_values, features=X_test_proc[:200], feature_names=numeric_features, show=False)
plt.tight_layout()
plt.savefig("shap_summary.png")
plt.show()

print("Artifacts saved in ./artifacts. SHAP summary in shap_summary.png")

# 12) Minimal prediction function for deployment
def predict_from_df(df_input):
    # df_input: raw dataframe with same columns as original X
    proc = joblib.load("artifacts/preprocessor.joblib")
    model_loaded = tf.keras.models.load_model("artifacts/house_price_model.keras")
    Xp = proc.transform(df_input)
    preds = model_loaded.predict(Xp).squeeze()
    # if you used log1p on target earlier, return np.expm1(preds)
    return preds

# Example usage:
# single_row = X_test.iloc[:3]
# print(predict_from_df(single_row))
