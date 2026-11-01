# Digit Recognizer

A very small AI project. Draw a digit from 0 to 9 and a neural network tells you which one it is.

To run the demo, download this repo and double-click `index.html`. Nothing needs installing and it works offline.

## How it works

1. **Data:** 1,797 images of handwritten digits, each 8x8 pixels (the digits dataset in scikit-learn).
2. **Split:** 80% of the images are used for training and 20% are kept aside for testing.
3. **Extra data:** each training image gets 10 varied copies (tilted, resized, moved, sharpened), so the model copes better with real drawings.
4. **Model:** a neural network with 64 inputs (one per pixel), 128 hidden neurons and 10 outputs (one per digit).
5. **Result:** 99.2% accuracy on the test images, which the model never saw in training.
6. **Demo:** the learned weights are saved to `model.js`, and `index.html` runs the model in the browser. Your drawing is cropped, centred and shrunk to 8x8 first, so it matches the training images.

## Known limit

The training images are only 8x8 pixels, so live drawings score lower than the test images. Some digits, especially 8 and 6, can be misread.

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

Python, scikit-learn, NumPy, HTML and JavaScript.
