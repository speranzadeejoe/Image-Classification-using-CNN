# Image-Classification-using-CNN
# Image Classification Using CNN

## Overview
This project implements an image classification model using Convolutional Neural Networks (CNNs) to classify images into 5 different categories. The dataset consists of **3,019 training images** and **1,298 testing images**. The model is trained using TensorFlow/Keras.

## Dataset
- **Dataset Source**: `/root/.cache/kagglehub/datasets/alxmamaev/flowers-recognition/versions/2/`
- **Classes**: 5
- **Training Images**: 3,019
- **Testing Images**: 1,298
- **Image Size**: 100x100 pixels

## Model Architecture
- **Convolutional Layers**: Extracts features from images.
- **Pooling Layers**: Reduces dimensionality and prevents overfitting.
- **Fully Connected Layers**: Maps features to class probabilities.
- **Optimizer**: Adam
- **Loss Function**: Categorical Crossentropy
- **Metrics**: Accuracy

## Training Process
```python
# Compile model
model.compile(loss="categorical_crossentropy", metrics=['accuracy'], optimizer='adam')

# Train model
history = model.fit(training_iterator, validation_data=testing_iterator, epochs=8)
```

## Issues & Fixes
### Observations:
- **Training Accuracy**: Improved from **25.8%** to **89.5%**.
- **Validation Accuracy**: Remains between **33.8% and 46.1%**.
- **Validation Loss Increases**, indicating **overfitting**.

### Fixes:
1. **Regularization**:
   - Add **Dropout layers** to prevent overfitting.
   - Apply **L2 regularization** to convolutional layers.
2. **Reduce Learning Rate**:
   - Use `Adam(learning_rate=0.0001)` for smoother training.
3. **Data Augmentation**:
   - Apply transformations (`rotation`, `zoom`, `flip`) to generalize better.
4. **Early Stopping**:
   - Stop training when validation loss increases.

## Installation & Usage
### Prerequisites
- Python 3.7+
- TensorFlow 2.x
- Keras
- NumPy, Pandas, Matplotlib

### Install Dependencies
```bash
pip install -r requirements.txt
```

### Run Training
```bash
python train_model.py
```

### Evaluate Model
```bash
python evaluate.py
```

## Results & Future Work
- The model currently **overfits** the training data.
- Future improvements:
  - **Increase dataset size**.
  - **Experiment with deeper architectures (ResNet, EfficientNet, etc.)**.
  - **Use Transfer Learning** for better generalization.

## License
This project is open-source and available under the MIT License.

