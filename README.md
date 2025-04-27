#Loan Approval Prediction
This project predicts loan approval using a Random Forest model and a Streamlit web application. It provides an interactive interface for users to input applicant details and view predictions with visualizations to understand the model's decision-making process.
Files

train_model.py: Script to train a Random Forest model on the Loan Prediction Problem Dataset and save it as loan_approval_model.pkl.
app.py: Streamlit app for entering applicant details, predicting loan approval, and displaying visualizations.
.gitignore: Excludes large files (loan_approval_model.pkl, train_u6lujuX_CVtuZ9i.csv) and Python cache/virtual environments.

Prerequisites

Python 3.8+
Git (for cloning the repository)
Required Python packages:streamlit
pandas
scikit-learn
joblib
plotly
matplotlib



Setup Instructions

Clone the Repository:
git clone https://github.com/your-username/PGM_PROJECT.git
cd PGM_PROJECT


Install Dependencies:
pip install streamlit pandas scikit-learn joblib plotly matplotlib


Download the Dataset:

Obtain train_u6lujuX_CVtuZ9i.csv from the Kaggle Loan Prediction Problem Dataset.
Place it in a local folder (e.g., C:\Users\YourUser\Downloads\archive (5)).
Update the TRAIN_DATA_PATH in train_model.py to point to the dataset, e.g.:TRAIN_DATA_PATH = r"C:\Users\YourUser\Downloads\archive (5)\train_u6lujuX_CVtuZ9i.csv"


Alternatively, place the dataset in the PGM_PROJECT folder and update train_model.py:import os
TRAIN_DATA_PATH = os.path.join(os.getcwd(), "train_u6lujuX_CVtuZ9i.csv")




Train the Model:

Run:python train_model.py


This generates loan_approval_model.pkl in the PGM_PROJECT folder.


Run the Streamlit App:
streamlit run app.py


Open the provided URL (e.g., http://localhost:8501) in a browser.



Usage

Input Applicant Details:

Use the Streamlit interface to enter details like Gender, Married, Dependents, Education, Self Employed, Applicant Income, Coapplicant Income, Loan Amount, Loan Amount Term, Credit History, and Property Area.
Click "Predict" to get the loan approval prediction (Approved or Denied) and confidence score.


Visualizations:

Decision Tree: Displays a sample tree from the Random Forest, showing decision splits (e.g., Credit_History <= 0.5).
Gauge Chart: Visualizes the prediction confidence (e.g., 92.5%) in a semi-circular gauge (green for Approved, red for Denied).
Feature Importance: Bar chart highlighting key features influencing predictions (e.g., Credit_History).



Notes

The dataset (train_u6lujuX_CVtuZ9i.csv) and model file (loan_approval_model.pkl) are excluded from the repository to reduce size. Follow the setup instructions to obtain the dataset and generate the model.
The Random Forest model is configured with max_depth=5 and min_samples_leaf=5 to ensure realistic confidence scores (e.g., 80-95%).
If the decision tree visualization is unreadable, adjust figsize or fontsize in app.py or reduce max_depth in train_model.py.

License
This project is licensed under the MIT License.
Acknowledgments

Dataset: Kaggle Loan Prediction Problem Dataset
Libraries: Streamlit, scikit-learn, Plotly, Matplotlib

