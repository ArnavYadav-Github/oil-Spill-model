
# 🛢️ Oil Spill Detection Model

This project focuses on identifying oil spills in marine environments using satellite or aerial images. It leverages deep learning techniques—specifically a U-Net architecture with a ResNet50 encoder—to perform semantic segmentation of images and accurately locate spill regions. By automating this process, it aims to support real-time environmental monitoring and enhance marine ecosystem protection.

---

## 📌 Key Features

- ✅ **Data Handling**: Loads and preprocesses satellite images and corresponding binary segmentation masks.
- 📸 **Visualization**: Displays raw images, preprocessed data, and labeled ground truth masks.
- 🧠 **Model Training**: Implements a CNN-based segmentation model using U-Net with a ResNet50 backbone.
- 📊 **Performance Metrics**: Evaluates the model using precision, recall, F1-score, confusion matrix, and accuracy.
- 🖼️ **Output Visualization**: Shows side-by-side visual comparisons of predicted masks vs. ground truth.
- ⚙️ **Deployability**: Designed for easy integration into web apps or real-time systems.

---

## 🧰 Project Requirements

> **Language**: Python 3.7+  
> **Development Environment**: Jupyter Notebook

### 🔧 Installation Steps

#### 1. Clone the Repository

```bash
git clone https://github.com/your-username/oil-spill-detection.git
cd oil-spill-detection
```

#### 2. Create and Activate Virtual Environment

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

#### 3. Install Required Dependencies

```bash
pip install -r requirements.txt
```

If `requirements.txt` is not available:

```bash
pip install numpy pandas matplotlib seaborn opencv-python scikit-learn tensorflow keras scikit-image tqdm
```

---

## 🔑 Libraries Overview

| Library           | Purpose                                      |
|-------------------|----------------------------------------------|
| `numpy`, `pandas` | Numerical computations and data handling     |
| `matplotlib`, `seaborn` | Visualization and exploratory analysis |
| `opencv-python`   | Image loading, resizing, and preprocessing   |
| `tensorflow`, `keras` | Model definition, training, and evaluation |
| `scikit-learn`    | Performance metrics and validation           |
| `scikit-image`    | Additional image transformation utilities    |
| `tqdm`            | Loop progress monitoring                     |

---

## 🗂️ Project Structure

```plaintext
.
├── oil-spill-detection-model.ipynb     # Main notebook
├── data/
│   ├── images/                         # Raw satellite images
│   └── masks/                          # Corresponding segmentation masks
├── models/                             # Directory for saved model weights
├── utils/                              # Helper functions (optional)
└── README.md                           # Project documentation
```

---

## ▶️ How to Use

### Step-by-Step Workflow

1. **Data Loading**
   - Reads satellite images from `data/images/`
   - Reads binary masks from `data/masks/`
   - Masks: `0` = no oil spill, `1` = oil spill

2. **Preprocessing**
   - Resizes all images and masks to a consistent shape (e.g., 128x128)
   - Normalizes image pixel values
   - Converts masks to categorical format if required

3. **Model Building**
   - Uses U-Net for semantic segmentation
   - Employs pre-trained ResNet50 as the encoder for better feature extraction
   - Allows inclusion of callbacks like `ModelCheckpoint` and `EarlyStopping`

4. **Model Training**
   - Splits data into training/validation sets
   - Trains over user-defined epochs
   - Monitors loss and accuracy metrics

5. **Evaluation**
   - Computes confusion matrix, precision, recall, F1-score
   - Generates training history plots (loss, accuracy)
   - Displays visual comparison of predictions vs. actual masks

---

## 📐 Configuration Notes

- Input Size: All images and masks are resized to 128x128 (modifiable).
- Mask Labels: Binary (0: background, 1: oil spill).
- Training Callbacks: Use `EarlyStopping`, `ReduceLROnPlateau`, `ModelCheckpoint` for optimal performance.

---

## 📊 Sample Output

- 📈 Accuracy/Loss Curves
- 📉 Confusion Matrix
- 🧠 Classification Report
- 🖼️ Visual output of predicted vs. actual masks

---

## 🚀 Future Scope

- 🌐 Integrate Sentinel Hub or Google Earth Engine APIs for real-time satellite image acquisition.
- 🔌 Build a web-based UI using Flask or Streamlit for interactive predictions.
- 📲 Package as a RESTful API for external integration.
- 🎯 Extend model for **multi-class segmentation** (e.g., oil spill, ships, land, clouds).
- 🔁 Experiment with architectures like U-Net++, DeepLabV3+, or Transformer-based models.

---

## 🤝 Contribution Guidelines

We welcome community contributions! To contribute:

```bash
# Step 1: Fork this repository
# Step 2: Create a feature branch
git checkout -b feature-your-feature-name

# Step 3: Make your changes and commit
git commit -m "Added new feature"

# Step 4: Push to GitHub
git push origin feature-your-feature-name

# Step 5: Open a Pull Request for review
```

---

## 📩 Contact & Support

For any questions or issues, feel free to open a GitHub Issue or reach out via email (replace with contact info).

---

