📂 Machine Learning & Deep Learning Lab Programs

This repository contains various implementations of Machine Learning (ML) and Deep Learning (DL) models on different datasets as part of lab exercises. Each program demonstrates a specific concept or algorithm with practical applications.

📜 Repository Structure & Descriptions

1️⃣ Artificial Neural Network (ANN) on Insurance Data



Task: Implemented an ANN model for predicting charges on the Insurance dataset.

Details:

Preprocessing: Encoding categorical features, scaling numerical data.

Model: ANN with multiple dense layers.

Loss Function: MSE (Regression task).

Optimizer: Adam.

Evaluation: MSE, MAE, and R² score.

Objective: Understand the basics of ANN for regression tasks.

2️⃣ ANN Model for Regression (Boston Housing Data)



Task: Built an ANN model to predict house prices using the Boston Housing dataset.

Details:

Data Preprocessing: Feature scaling using StandardScaler.

ANN Architecture: Input → Hidden layers → Output.

Activation: ReLU, Linear.

Loss Function: MSE.

Evaluation: RMSE, R² score.

Objective: Gain experience with regression modeling using deep learning.

3️⃣ Perceptron Learning Rule with Different Activation Functions


Task: Implemented the Perceptron learning algorithm on the Insurance dataset with different activation functions like:

Step Function

Sigmoid

Tanh

Objective: Compare how different activation functions affect the perceptron model's performance.

4️⃣ MLP with Dropout and Batch Normalization (PCOS Dataset)


Task: Designed and trained a Multi-Layer Perceptron (MLP) for classification on the PCOS dataset with:

Dropout layers to prevent overfitting.

Batch Normalization to stabilize training.

Metrics: Accuracy, Precision, Recall, F1-score.

Objective: Learn regularization techniques in deep learning.

5️⃣ Convolution Operations & Feature Map Analysis (CIFAR-10 Dataset)


Task:

Implemented basic convolution operations from scratch.

Visualized feature maps to analyze what CNN filters learn.

Dataset: CIFAR-10 (Image classification dataset).

Objective: Understand how convolution layers extract features in image data.

6️⃣ LSTM Model on Daily Delhi Climate Data



Task: Built an LSTM (Long Short-Term Memory) model for time series forecasting of Daily Delhi Climate data.

Objective:

Learn sequential modeling with LSTMs.

Predict future temperature values based on past data.

7️⃣ GRU Model on Air Passengers Data


Task: Implemented a GRU (Gated Recurrent Unit) model for Air Passengers dataset to predict passenger traffic.

Objective: Compare GRU performance with LSTM for time series forecasting tasks.

🛠️ Tools & Libraries Used

Programming Language: Python

Libraries:

numpy, pandas, matplotlib, seaborn for data handling & visualization

scikit-learn for preprocessing & evaluation

tensorflow, keras, pytorch (if used) for deep learning models

📊 Key Learning Outcomes

Implemented different ML & DL algorithms for regression, classification, and time series forecasting tasks.

Understood the role of activation functions, dropout, batch normalization, and convolution layers.

Worked with real-world datasets for practical applications.

Gained hands-on experience with feature engineering, model training, and evaluation metrics.

📁 Datasets Used

Insurance Dataset (Regression)

Boston Housing Dataset (Regression)

PCOS Dataset (Classification)

CIFAR-10 Dataset (Image Classification)

Daily Delhi Climate Dataset (Time Series)

Air Passengers Dataset (Time Series)

📌 Future Scope

Hyperparameter tuning for better accuracy.

Experimenting with advanced architectures (e.g., ResNet, Transformers).

Deployment of trained models as web apps or APIs.
