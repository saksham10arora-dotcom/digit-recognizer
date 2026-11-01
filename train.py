"""Train a small neural network to read handwritten digits, then save it for the web demo."""
import json

import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier

# 1. Load MNIST: 70,000 images of handwritten digits, each 28x28 pixels.
# The first run downloads it into data/, later runs reuse that copy.
X, y = fetch_openml("mnist_784", version=1, as_frame=False, data_home="data", return_X_y=True)
X = X / 255  # pixel values go from 0-255, scale them to 0-1
y = y.astype(int)

# 2. Use the standard split. The model never sees the last 10,000 images during training.
X_train, X_test = X[:60000], X[60000:]
y_train, y_test = y[:60000], y[60000:]

# 3. Train: 784 input pixels -> 128 hidden neurons -> 10 outputs (one per digit).
model = MLPClassifier(hidden_layer_sizes=(128,), max_iter=20, random_state=42)
model.fit(X_train, y_train)

# 4. Test on the images the model has never seen.
accuracy = accuracy_score(y_test, model.predict(X_test))
print(f"Training images: {len(X_train)}")
print(f"Test images:     {len(X_test)}")
print(f"Test accuracy:   {accuracy:.1%}")

# 5. Save the learned weights so the web page can run the model in the browser.
weights = {
    "hidden_weights": np.round(model.coefs_[0], 3).tolist(),
    "hidden_bias": np.round(model.intercepts_[0], 3).tolist(),
    "output_weights": np.round(model.coefs_[1], 3).tolist(),
    "output_bias": np.round(model.intercepts_[1], 3).tolist(),
    "accuracy": round(accuracy * 100, 1),
}
with open("model.js", "w") as f:
    f.write("const MODEL = " + json.dumps(weights, separators=(",", ":")) + ";\n")
print("Saved model.js")
