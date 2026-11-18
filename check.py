import pickle
import numpy as np
import matplotlib.pyplot as plt
import keras
from sklearn.metrics import roc_auc_score, mean_squared_error, mean_absolute_error

OUTPUT_DIR = "forecasting/Results"

# 1. Load Model and Test Data
model = keras.saving.load_model(f"{OUTPUT_DIR}/model.keras", compile=False)
with open(f"{OUTPUT_DIR}/X_test.p", "rb") as f:
    X_test = pickle.load(f)
with open(f"{OUTPUT_DIR}/y_test.p", "rb") as f:
    y_test = pickle.load(f)

# Ensure float32 inputs for keras
if isinstance(X_test, list):
    X_test = [np.asarray(x, dtype=np.float32) for x in X_test]
y_test = np.asarray(y_test, dtype=np.float32)

y_pred = model.predict(X_test)

# 2. Regression Metrics
wape = np.sum(np.abs(y_test - y_pred)) / (np.sum(np.abs(y_test)) + 1e-8)
rmse = np.sqrt(mean_squared_error(y_test.ravel(), y_pred.ravel()))
mae = mean_absolute_error(y_test.ravel(), y_pred.ravel())

# 3. AUROC Metrics
# A. Global AUROC (Any Call: y > 0)
y_bin_any = (y_test.ravel() > 0).astype(int)
y_scores = y_pred.ravel()

if len(np.unique(y_bin_any)) > 1:
    global_auroc = roc_auc_score(y_bin_any, y_scores)
else:
    global_auroc = float("nan")

# B. High-Demand AUROC (e.g. calls >= 2)
threshold = 2.0
y_bin_high = (y_test.ravel() >= threshold).astype(int)
if len(np.unique(y_bin_high)) > 1:
    high_demand_auroc = roc_auc_score(y_bin_high, y_scores)
else:
    high_demand_auroc = float("nan")

# C. Spatial Cell-Averaged AUROC (measures ranking per cell across time)
cell_aurocs = []
nr_rows, nr_cols = y_test.shape[1], y_test.shape[2]
for r in range(nr_rows):
    for c in range(nr_cols):
        cell_y = (y_test[:, r, c] > 0).astype(int)
        cell_pred = y_pred[:, r, c]
        if len(np.unique(cell_y)) > 1:
            cell_aurocs.append(roc_auc_score(cell_y, cell_pred))
mean_cell_auroc = np.nanmean(cell_aurocs) if cell_aurocs else float("nan")

# 4. Print Summary Table
print("=" * 55)
print("              EVALUATION METRICS SUMMARY             ")
print("=" * 55)
print(f"  Test WAPE:               {wape:.4f} ({wape*100:.2f}%)")
print(f"  Test RMSE:               {rmse:.4f}")
print(f"  Test MAE:                {mae:.4f}")
print("-" * 55)
print(f"  Global AUROC (calls > 0): {global_auroc:.4f}")
print(f"  High-Demand AUROC (>= {threshold:.0f}): {high_demand_auroc:.4f}")
print(f"  Mean Cell AUROC:         {mean_cell_auroc:.4f}")
print("=" * 55)

# 5. Spatial Visualization
fig, axes = plt.subplots(1, 3, figsize=(14, 4))
axes[0].imshow(y_test.mean(axis=0), cmap="YlOrRd")
axes[0].set_title("Mean Actual Demand")
axes[1].imshow(y_pred.mean(axis=0), cmap="YlOrRd")
axes[1].set_title("Mean Predicted Demand")
im = axes[2].imshow((y_test - y_pred).mean(axis=0), cmap="coolwarm")
axes[2].set_title("Mean Residual (Actual - Pred)")
plt.colorbar(im, ax=axes[2])
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/spatial_error_map.png", dpi=200)
print(f"Spatial residual map saved to {OUTPUT_DIR}/spatial_error_map.png")