"""Train a small neural network to read handwritten digits, then save it for the web demo."""
import json

import numpy as np
from scipy import ndimage
from sklearn.datasets import load_digits
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

# 1. Load the data: 1,797 images of digits, each 8x8 pixels. It ships with scikit-learn.
digits = load_digits()
X = digits.data / 16  # pixel values go from 0-16, scale them to 0-1
y = digits.target

# 2. Keep 20% of the images aside. The model never sees these during training.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)


# 3. Add varied copies of each training image. Real drawings are tilted, off-centre
# and have sharper lines than this dataset, so the model needs to see that too.
rng = np.random.default_rng(42)


def vary(image):
    big = ndimage.zoom(image.reshape(8, 8), 4, order=1)  # work at 32x32
    angle, zoom = rng.uniform(-0.2, 0.2), rng.uniform(0.85, 1.15)
    cos, sin = np.cos(angle) / zoom, np.sin(angle) / zoom
    matrix = np.array([[cos, -sin], [sin, cos]])
    offset = 16 - matrix @ [16, 16] + rng.uniform(-3, 3, 2)
    big = ndimage.affine_transform(big, matrix, offset=offset, order=1)
    big = np.clip((big - rng.uniform(0.25, 0.5)) * rng.uniform(2.5, 5), 0, 1)  # sharpen
    return big.reshape(8, 4, 8, 4).mean(axis=(1, 3)).ravel()  # back to 8x8


COPIES = 10
varied = np.array([vary(image) for image in X_train for _ in range(COPIES)])
X_train = np.vstack([X_train, varied])
y_train = np.concatenate([y_train, np.repeat(y_train, COPIES)])

# 4. Train: 64 input pixels -> 128 hidden neurons -> 10 outputs (one per digit).
model = MLPClassifier(hidden_layer_sizes=(128,), max_iter=300, random_state=42)
model.fit(X_train, y_train)

# 5. Test on the images the model has never seen.
accuracy = accuracy_score(y_test, model.predict(X_test))
print(f"Training images: {len(X_train)}")
print(f"Test images:     {len(X_test)}")
print(f"Test accuracy:   {accuracy:.1%}")

# 6. Save the learned weights so the web page can run the model in the browser.
weights = {
    "hidden_weights": np.round(model.coefs_[0], 4).tolist(),
    "hidden_bias": np.round(model.intercepts_[0], 4).tolist(),
    "output_weights": np.round(model.coefs_[1], 4).tolist(),
    "output_bias": np.round(model.intercepts_[1], 4).tolist(),
    "accuracy": round(accuracy * 100, 1),
}
with open("model.js", "w") as f:
    f.write("const MODEL = " + json.dumps(weights, separators=(",", ":")) + ";\n")
print("Saved model.js")
