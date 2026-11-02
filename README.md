#  CNN from Scratch – CIFAR-10 Image Classification

### Deep Learning | PR-4 | Computer Vision Project

---

## 1. Project Overview

This project focuses on building and evaluating Convolutional Neural Network (CNN) models for image classification using the **CIFAR-10 dataset**.

The main objective is to understand how deep learning models identify visual patterns, extract important image features, and classify images into different categories.

Throughout this project, I implemented CNN architectures from scratch, explored convolution and pooling operations, applied data augmentation techniques, and experimented with transfer learning using MobileNetV2.

I also compared six different deep learning models to understand their performance and classification capabilities.

##  2. Project Objectives

The main objectives of this project are:

- Understand the fundamentals of Convolutional Neural Networks.
- Load, explore, and preprocess the CIFAR-10 dataset.
- Understand manual convolution operations and feature extraction.
- Implement different pooling techniques.
- Build and train a LeNet-5 architecture.
- Develop a custom CNN model from scratch.
- Understand backpropagation and model optimization.
- Apply data augmentation to improve model generalization.
- Implement transfer learning using MobileNetV2.
- Compare different models using test accuracy and loss.
- Evaluate model performance using confusion matrices and visualization techniques.

##  3. Dataset Information

This project uses the **CIFAR-10 dataset**, which is a widely used dataset for image classification tasks.

| Dataset Details | Information |
|---|---|
| Dataset Name | CIFAR-10 |
| Total Images | 60,000 |
| Training Images | 50,000 |
| Testing Images | 10,000 |
| Image Dimensions | 32 × 32 |
| Image Channels | 3 (RGB) |
| Number of Classes | 10 |
| Image Type | Colour Images |

### CIFAR-10 Classes

The dataset contains the following ten categories:

1. Airplane
2. Automobile
3. Bird
4. Cat
5. Deer
6. Dog
7. Frog
8. Horse
9. Ship
10. Truck

### Dataset Loading

The dataset was loaded directly using TensorFlow/Keras:

```python
import tensorflow as tf

(x_train, y_train), (x_test, y_test) = (
    tf.keras.datasets.cifar10.load_data()
)
```

##  4. Technologies and Libraries Used

| Technology | Purpose |
|---|---|
| Python | Programming language |
| Google Colab | Development environment |
| TensorFlow | Deep learning framework |
| Keras | Neural network implementation |
| NumPy | Numerical computations |
| Pandas | Data handling |
| Matplotlib | Data visualization |
| Scikit-learn | Model evaluation |
| SciPy | Manual convolution operations |

##  5. Project Implementation

The project was divided into six major stages.

### A1 – Data Loading and Exploratory Data Analysis

- Loaded the CIFAR-10 dataset.
- Checked dataset shapes and dimensions.
- Displayed sample images from all ten classes.
- Analyzed pixel intensity distributions.
- Normalized image pixel values between 0 and 1.
- Converted class labels into one-hot encoded format.
- Studied the difference between CNN and traditional ANN.

### A2 – Convolution Operations and Feature Extraction

- Converted sample RGB images into grayscale.
- Implemented manual convolution using a 3×3 edge detection kernel.
- Explored VALID and SAME padding.
- Compared stride 1 and stride 2.
- Applied multiple filters to understand feature extraction.
- Visualized the resulting feature maps.

### A3 – Pooling and CNN Architectures

Implemented and explored:

**1. Max Pooling**

Selects the maximum value from each pooling region.

**2. Average Pooling**

Calculates the average value from each pooling region.

**3. LeNet-5 Architecture**

Implemented a LeNet-5-style architecture using convolutional layers, average pooling, fully connected layers, and softmax classification.

**4. Custom CNN Architecture**

Developed a modern CNN with:

- Convolutional layers
- ReLU activation
- Max pooling
- Global average pooling
- Fully connected layers
- Softmax output layer

The custom CNN was designed to remain below the 2-million-parameter requirement.

### A4 – Model Training and Evaluation

- Studied the backpropagation algorithm.
- Compiled the CNN using the Adam optimizer.
- Used categorical cross-entropy as the loss function.
- Trained the model using training and validation datasets.
- Applied EarlyStopping.
- Applied ReduceLROnPlateau.
- Visualized training and validation accuracy.
- Visualized training and validation loss.
- Evaluated the model using test data.
- Generated a confusion matrix.
- Displayed incorrectly classified images.

### A5 – Data Augmentation

Explored data augmentation techniques to increase training image variation.

**ImageDataGenerator techniques:**

- Rotation
- Width shifting
- Height shifting
- Horizontal flipping
- Zooming

**Keras augmentation layers:**

- RandomFlip
- RandomRotation
- RandomTranslation
- RandomZoom

Trained separate CNN models using these augmentation approaches and compared their performance with the original CNN.

### A6 – Transfer Learning with MobileNetV2

Implemented transfer learning using the pretrained MobileNetV2 architecture.

**Feature Extraction:**

- Loaded MobileNetV2 with ImageNet pretrained weights.
- Resized images to 96 × 96 pixels.
- Froze the pretrained convolutional layers.
- Added Global Average Pooling.
- Added fully connected classification layers.
- Trained the new classification layers.

**Fine-Tuning:**

- Unfroze the last 30 layers of the pretrained model.
- Used a lower learning rate.
- Continued model training to adapt the pretrained features to CIFAR-10.

Also implemented and demonstrated a residual connection block to understand skip connections in deep neural networks.

##  6. Models Compared

Six models were evaluated using the CIFAR-10 test dataset.

| No. | Model |
|---|---|
| 1 | LeNet-5 |
| 2 | Custom CNN without augmentation |
| 3 | CNN with ImageDataGenerator |
| 4 | CNN with Keras Augmentation |
| 5 | MobileNetV2 Feature Extraction |
| 6 | MobileNetV2 Fine-Tuning |

### Model Performance Comparison

The following table should be updated with the actual test results obtained from Google Colab.

| Model | Test Accuracy |
|---|---|
| LeNet-5 | Add actual result |
| Original CNN | Add actual result |
| ImageDataGenerator CNN | Add actual result |
| Keras Augmentation CNN | Add actual result |
| MobileNetV2 Feature Extraction | Add actual result |
| MobileNetV2 Fine-Tuning | Add actual result |

### Performance Visualization

The project includes visual comparisons of all six models using Matplotlib.

![Final Model Comparison](plots/final_model_comparison.png)

## 7. Project Visualizations

The following visualizations are included in the project:

- CIFAR-10 sample images.
- Pixel intensity histograms.
- Manual convolution outputs.
- Feature map visualizations.
- Max pooling and average pooling comparison.
- Data augmentation examples.
- Training and validation accuracy graphs.
- Training and validation loss graphs.
- Confusion matrix.
- Misclassified image samples.
- Final six-model accuracy comparison.

##  8. Project Folder Structure

```text
PR4-CNN-CIFAR10/
│
├── DL_PR4A.ipynb
├── DL_PR4A.html
├── README.md
├── requirements.txt
│
├── plots/
│   ├── dataset_samples.png
│   ├── pixel_histogram.png
│   ├── feature_maps.png
│   ├── augmentation.png
│   ├── training_accuracy.png
│   ├── training_loss.png
│   ├── confusion_matrix.png
│   └── final_model_comparison.png
│
└── presentation/
    └── project_video.mp4
```

*Note: The folder structure represents the recommended submission organization. Include only files that you have actually generated.*

## 9. Installation and Setup

### Step 1 – Clone the Repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
```

### Step 2 – Open the Project Folder

```bash
cd PR4-CNN-CIFAR10
```

### Step 3 – Install Required Libraries

```bash
pip install -r requirements.txt
```

### Step 4 – Run the Notebook

Open `DL_PR4A.ipynb` using Google Colab or Jupyter Notebook.

The CIFAR-10 dataset is downloaded automatically through TensorFlow/Keras when the dataset loading cell is executed.

## 10. Requirements

The main libraries required for this project are:

```text
tensorflow
numpy
pandas
matplotlib
scikit-learn
scipy
```

##  11. Learning Outcomes

After completing this project, I gained practical knowledge of:

- Understanding CNN architecture and its working principles.
- Image preprocessing and normalization.
- Convolution, padding, stride, and pooling operations.
- Feature extraction from images.
- Building and training neural networks.
- Understanding backpropagation and optimization.
- Applying data augmentation techniques.
- Using pretrained models for transfer learning.
- Evaluating deep learning models.
- Creating performance graphs and visualizations.
- Working with TensorFlow and Keras in Google Colab.

##  12. Conclusion

Through this project, I learned how Convolutional Neural Networks work and how they can be used for image classification. I started with data loading and preprocessing, explored convolution and pooling operations, and gradually developed different CNN architectures.

I also experimented with data augmentation and transfer learning using MobileNetV2. Comparing six different models helped me understand how different architectures and training techniques affect classification performance.

Overall, this project gave me valuable hands-on experience in deep learning and computer vision. It helped me connect theoretical concepts with practical implementation and improved my confidence in developing image classification models.

##  13. Future Improvements

Some possible improvements for this project include:

- Experimenting with deeper CNN architectures.
- Using ResNet-based models.
- Applying more advanced optimization techniques.
- Improving classification accuracy through hyperparameter tuning.
- Exploring additional data augmentation methods.
- Testing the model on other image classification datasets.
- Developing a simple web application for image prediction.

## 14. Author Name : Swarna Ajay pathak 
Project demonstration video :(https://drive.google.com/file/d/1MM_E92PbsncYnUaO7x1FRe2E8khckIqs/view?usp=sharing)


