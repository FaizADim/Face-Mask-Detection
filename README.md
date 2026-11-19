# Face Mask Detection

A PyTorch object-detection project that identifies faces in images and classifies whether a person is:

- Wearing a mask
- Not wearing a mask
- Wearing a mask incorrectly

The project uses a Faster R-CNN model with a ResNet-50 FPN backbone and an annotated image dataset. The complete workflow is implemented in `face-mask-detection.ipynb`.

## Dataset

The `archive` directory contains 853 images and matching XML annotations:

```text
archive/
├── images/
│   └── maksssksksss*.png
└── annotations/
    └── maksssksksss*.xml
```

Each XML file contains Pascal VOC-style face bounding boxes and one of three labels:

| Label | Class ID |
| --- | --- |
| `with_mask` | 1 |
| `without_mask` | 2 |
| `mask_weared_incorrect` | 3 |

Class ID `0` is reserved for the background by Faster R-CNN.

## Model

The notebook loads the pretrained `fasterrcnn_resnet50_fpn` model from torchvision and replaces its box predictor with a four-class predictor:

- One background class
- Three mask classes

The model is currently configured for CPU training. It also contains detection code that can be used from the notebook after training.

## Requirements

- Python 3
- Jupyter Notebook or JupyterLab
- PyTorch
- torchvision
- NumPy
- Pillow
- Matplotlib
- tqdm

## Installation

Clone the repository and enter the project directory:

```bash
git clone https://github.com/FaizADim/Face-Mask-Detection.git
cd Face-Mask-Detection
```

Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install Jupyter and the required packages:

```bash
python3 -m pip install --upgrade pip
python3 -m pip install jupyter torch torchvision numpy pillow matplotlib tqdm
```

## Usage

Start Jupyter:

```bash
jupyter notebook
```

Open `face-mask-detection.ipynb` and run the cells from top to bottom.

Before running the training cells, update the dataset path in the notebook if this repository is located somewhere other than the default:

```python
BASE_DIR = "/Users/"Username"/Github/Face-Mask-Detection/archive"
```

For example, use a path relative to the notebook:

```python
BASE_DIR = "./archive"
```

The notebook:

1. Parses bounding boxes and labels from XML annotations.
2. Loads paired PNG images and annotations with `FaceMaskDataset`.
3. Initializes Faster R-CNN for the three mask classes.
4. Trains the model for one epoch by default.
5. Runs a prediction on `archive/images/maksssksksss0.png`.
6. Displays detected boxes with a configurable confidence threshold.

To test another image, change the `img_path` variable in the **Test Image** section:

```python
img_path = f"{IMG_DIR}/maksssksksss0.png"
image, pred = predict(img_path)
show(image, pred)
```

## Training Notes

- The default batch size is `4`.
- The default optimizer is SGD with a learning rate of `0.005`, momentum of `0.9`, and weight decay of `0.0005`.
- The notebook uses one training epoch by default. You can change this part
- The model and predictions remain in memory unless you add code to save and load model checkpoints.


