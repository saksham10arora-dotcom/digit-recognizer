# Digit Recognizer

A very small AI project. Draw a digit from 0 to 9 and a neural network tells you which one it is.

**Live demo:** https://saksham10arora-dotcom.github.io/digit-recognizer/

To run the demo without internet, download this repo and double-click `index.html`. Nothing needs installing.

## How it works

1. **Data:** MNIST, 70,000 images of handwritten digits, each 28x28 pixels.
2. **Split:** 60,000 images are used for training and 10,000 are kept aside for testing.
3. **Model:** a neural network with 784 inputs (one per pixel), 128 hidden neurons and 10 outputs (one per digit).
4. **Result:** 97.8% accuracy on the 10,000 test images, which the model never saw in training.
5. **Demo:** the learned weights are saved to `model.js`, and `index.html` runs the model in the browser. Your drawing is fitted into a 20x20 box and centred in a 28x28 image first, the same way MNIST was prepared.

## Tips for drawing

Draw one large digit. Give a 6 a long tail and a small loop, because a 6 with a big round loop can be read as a 0.

## Files

| File | What it does |
|------|--------------|
| `train.py` | Loads the data, trains the model, prints accuracy, saves `model.js` |
| `model.js` | The trained model weights |
| `index.html` | The drawing demo |

## Train it yourself

```bash
pip install -r requirements.txt
python train.py
```

## Tools used

Python, scikit-learn, NumPy, HTML and JavaScript. The first training run downloads MNIST (about 15 MB).
