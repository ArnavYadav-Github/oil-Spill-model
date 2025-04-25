# 🛢️ Oil Spill Detection Model

This project aims to detect oil spills from satellite images using image processing and deep learning techniques. The goal is to aid real-time environmental monitoring and support marine ecosystem protection by automating the detection of oil spill regions.

---

## 📌 Features

- ✅ Loads and preprocesses satellite or aerial image data
- 📸 Visualizes input images and ground-truth masks
- 🧠 Trains a CNN-based model (with U-Net architecture and ResNet50 encoder)
- 📈 Evaluates model performance using precision, recall, F1 score, and confusion matrix
- 🖼️ Visual comparison of predicted and actual oil spill masks
- ⚙️ Can be extended for real-time deployment or wrapped in a web app

---

## 🧰 Requirements

This project is implemented in **Python** using **Jupyter Notebook**.

### ✅ Python Dependencies

To install all dependencies:
pip install -r requirements.txt

If requirements.txt is not available, manually install the required libraries:
pip install numpy pandas matplotlib seaborn opencv-python scikit-learn tensorflow keras scikit-image tqdm

🔑 Key Libraries Used

Library	Purpose
numpy, pandas	Numerical operations & data handling
matplotlib, seaborn	Visualization and plotting
opencv-python	Image loading and preprocessing
tensorflow, keras	Deep learning model architecture
scikit-learn	Evaluation metrics
skimage	Image transformation tools
tqdm	Progress bar for loops


🗂️ Project Structure
.
├── oil-spill-detection-model.ipynb     # Main Jupyter Notebook
├── data/
│   ├── images/                         # Input satellite images
│   └── masks/                          # Corresponding segmentation masks
├── models/                             # (Optional) Directory to save trained models
└── README.md                           # Project documentation


▶️ How to Run the Project
1. Clone the Repository
git clone https://github.com/your-username/oil-spill-detection.git
cd oil-spill-detection

3. Launch Jupyter Notebook
jupyter notebook

5. Open the Notebook
Open oil-spill-detection-model.ipynb

Run all cells in order to:
✅ Load and preprocess the image and mask data
🏗️ Build the CNN model
🏋️ Train and validate the model
📊 Evaluate the model’s predictions
🖼️ Visualize the outputs
⚙️ Configuration Notes
📐 All images and masks are resized to a consistent shape (e.g., 128x128)


🏷️ Labels for segmentation masks should be binary:
0 = No Oil Spill
1 = Oil Spill

💾 You can use ModelCheckpoint or EarlyStopping callbacks for better training control

🔁 Supports switching between classical ML and CNN-based deep learning models

🧠 The model uses a pre-trained ResNet50 as an encoder in a U-Net architecture

📊 Model Output
The notebook produces:
✅ Training and validation accuracy/loss plots
📉 Confusion matrix
🧠 Precision, Recall, F1-score metrics
🖼️ Side-by-side comparison of predicted vs actual segmentation masks


🚀 Future Improvements
🔌 Add Flask or Streamlit web interface for interactive predictions
🌐 Integrate satellite APIs (e.g., Sentinel Hub) for real-time image input
📦 Extend to multi-class segmentation (e.g., ships, land, spill, clouds)
📲 Deploy as REST API for integration into external systems
🧠 Use transfer learning with advanced architectures like DeepLabV3+, U-Net++

🤝 Contribution
We welcome contributions! To contribute:
🍴 Fork the repository
🛠️ Create a new branch:
git checkout -b feature-xyz
💾 Commit your changes
🚀 Push the branch:
git push origin feature-xyz
📩 Create a Pull Request



