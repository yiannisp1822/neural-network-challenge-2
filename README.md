# neural-network-challenge-2
Yiannis Pagkalos  

# Employee Attrition Prediction Model

## Overview
This project aims to predict employee attrition and department classification using a deep learning model implemented in TensorFlow/Keras. The dataset contains various employee attributes, and the model leverages a multi-output neural network to make predictions.

## Dataset
The dataset is sourced from a CSV file and includes features such as:
- Age
- DistanceFromHome
- Education
- OverTime
- StockOptionLevel
- WorkLifeBalance
- YearsAtCompany
- JobSatisfaction
- NumCompaniesWorked
- YearsSinceLastPromotion

The target variables are:
1. **Attrition** (Binary Classification: Yes/No)
2. **Department** (Multi-Class Classification: HR, R&D, Sales)

## Model Architecture
The model is a multi-output deep neural network with the following layers:
- **Input Layer**: Accepts numerical features.
- **Shared Hidden Layers**: Two dense layers with ReLU activation.
- **Branch for Attrition Prediction**:
  - Hidden layer with 32 neurons and ReLU activation.
  - Output layer with 2 neurons and softmax activation.
- **Branch for Department Prediction**:
  - Hidden layer with 32 neurons and ReLU activation.
  - Output layer with 3 neurons and softmax activation.

The model is compiled using the **Adam optimizer** and **categorical cross-entropy loss** for both outputs. Accuracy is used as the evaluation metric.

## Installation & Dependencies
To run this project, install the required dependencies:
```
conda install tensorflow pandas numpy scikit-learn
```

## Usage
1. Load and preprocess the dataset.
2. Train the model using:
   ```python
   model.fit(X_train_scaled, [y_train_attrition, y_train_department], epochs=50, batch_size=32)
   ```
3. Evaluate the model:
   ```python
   model.evaluate(X_test_scaled, [y_test_attrition, y_test_department])
   ```

## Potential Improvements
- Address class imbalance using resampling or weighted loss.
- Use different activation functions or optimizers.
- Perform feature engineering to include additional relevant attributes.
- Tune hyperparameters for better performance.

## License
This project is open-source and available for modification and use under the MIT License.

