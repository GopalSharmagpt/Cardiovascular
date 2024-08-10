# Cardiovascular

# Cardiovascular Disease Classification

This project involves building a machine learning model to classify patients as healthy or suffering from cardiovascular disease based on various attributes. The model is implemented using Python and leverages the Random Forest algorithm for classification.

## Dataset

The dataset used in this project is `cardio.csv`, which contains the following features:

- **age**: Age of the patient in days (converted to years in the preprocessing step)
- **height**: Height of the patient in cm
- **weight**: Weight of the patient in kg
- **ap_hi**: Systolic blood pressure
- **ap_lo**: Diastolic blood pressure
- **cholesterol**: Cholesterol level (1: normal, 2: above normal, 3: well above normal)
- **gluc**: Glucose level (1: normal, 2: above normal, 3: well above normal)
- **smoke**: Smoking status (0: non-smoker, 1: smoker)
- **alco**: Alcohol intake (0: non-drinker, 1: drinker)
- **active**: Physical activity (0: low, 1: high)
- **cardio**: Target variable (0: healthy, 1: suffering from cardiovascular disease)

## Project Structure

- **cardio.csv**: The dataset used for training and testing the model.
- **cardio_classification.py**: The Python script containing the implementation of the model.
- **README.md**: This file, providing an overview of the project.

## Installation

To run this project, ensure you have the following packages installed:

```bash
pip install pandas numpy scikit-learn matplotlib
