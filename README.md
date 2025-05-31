# Plant Species Identifier – Campus Edition

An AI-powered image classification system designed to identify various plant species within our college campus using deep learning.

---

##  Project Description and Objectives

### Description

This project is a practical implementation of computer vision using convolutional neural networks (CNNs) to identify plant species found on our college campus. By collecting, training, and evaluating real-world plant images, we created a model capable of classifying species from new images.

### Objectives

- Build and evaluate multiple deep learning models to classify plant species.
- Tune hyperparameters (optimizer, learning rate, batch size, decay strategies) to improve model accuracy.
- Structure the code modularly for ease of development and maintenance.
- Deploy the trained model using both Gradio (HuggingFace Spaces) and Flask.

---

## Installation and Setup Instructions

### Step 1: Clone the Repository

git clone https://github.com/SangayZin/02230298_DAM101_Final_Project
cd plant-species-campus


### Step 2: Create and Activate a Virtual Environment

python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate



### Step 3: Install Required Dependencies

pip install -r requirements.txt

Step 4: Dataset Structure

### Organize your dataset as:

/data
  ├── train/
  │   ├── species_1/
  │   ├── species_2/
  │   └── ...
  ├── valid/
  └── test/

Each folder should contain JPEG/PNG images of one plant species.

### ⚙️ Usage Examples

#### To Train the Model

    python train.py

#### To Evaluate the Model

    python evaluate.py

#### To Launch the Gradio Interface

    python gradio_app.py

#### Expected output:

A web interface opens where users can upload an image and receive the predicted species.

#### To Launch the Flask Web App

    python app.py

Expected output:

Access the local Flask app at http://localhost:5000 to upload and classify images.

###  Data Preparation Guidelines

- Images of plants were manually collected using mobile devices across various campus zones.

- Dataset was labeled manually with plant species names.

 - Images were resized to 224x224 and augmented using transformations like:

        - RandomCrop, RandomHorizontalFlip, ColorJitter, and Normalization

- Data split:

    - 70% Training

    - 20% Validation

    - 10% Testing

###  Model Architecture Details

We implemented and evaluated three models:

### 1. Baseline CNN

  Basic Conv2D + ReLU + MaxPool layers

  Served as the initial benchmark

### 2. Deep CNN (Best Model)

  3 Conv2D blocks with BatchNorm, Dropout

   Flatten → Dense layers

   Regularization to avoid overfitting

### 3. Transfer Learning

  Based on pre-trained models (e.g., ResNet50/EfficientNet)

   Fine-tuned only top layers

### Best Hyperparameters for Deep CNN

| Hyperparameter     | Value                   |
|--------------------|--------------------------|
| Optimizer          | Adam                    |
| Learning Rate      | 0.0005                  |
| Batch Size         | 32                      |
| Epochs             | 20                      |
| Scheduler          | Cosine Annealing        |
| Loss Function      | CrossEntropyLoss        |
| Augmentation       | RandomCrop, Flip, Jitter |


 ###  Performance Metrics and Evaluation Results

| Model              | Validation Accuracy | Test Accuracy |
|-------------------|---------------------|---------------|
| Baseline CNN      | 72.4%               | 70.1%         |
| Deep CNN (Best)   | **89.3%**           | **88.1%**     |
| Transfer Learning | 87.5%               | 86.2%         |
   
   
### Confusion Matrix

- Available in results/confusion_matrix.png

- High classification accuracy across dominant species

- Minor confusion between visually similar species

### TensorBoard Visuals

Track training/validation loss and accuracy in real time:

    tensorboard --logdir=runs

TensorBoard screenshots are included in results/tensorboard/.
