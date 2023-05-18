import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix,roc_curve,auc, roc_auc_score
#stage1
# Load the data set from the CSV file
df = pd.read_csv('/Users/mertatalay/Desktop/programacionII/challange/data.csv')

# Define the descriptions for each attribute
descriptions = {
    'diagnosis': 'The diagnosis of breast tissues (M = malignant, B = benign)',
    'radius_mean': 'Mean of distances from center to points on the perimeter',
    'texture_mean': 'Standard deviation of gray-scale values',
    'perimeter_mean': 'Mean size of the core tumor',
    'area_mean': 'Mean area of the tumor',
    'smoothness_mean': 'Mean smoothness of the tumor',
    'compactness_mean': 'Mean compactness of the tumor',
    'concavity_mean': 'Mean severity of concave portions of the contour',
    'concave points_mean': 'Mean number of concave portions of the contour',
    'symmetry_mean': 'Mean symmetry of the tumor',
    'fractal_dimension_mean': 'Mean "coastline approximation" of the tumor',
    'radius_se': 'Standard error of distances from center to points on the perimeter',
    'texture_se': 'Standard error of gray-scale values',
    'perimeter_se': 'Standard error of the core tumor size',
    'area_se': 'Standard error of the tumor area',
    'smoothness_se': 'Standard error of the tumor smoothness',
    'compactness_se': 'Standard error of the tumor compactness',
    'concavity_se': 'Standard error of the severity of concave portions of the contour',
    'concave points_se': 'Standard error of the number of concave portions of the contour',
    'symmetry_se': 'Standard error of the tumor symmetry',
    'fractal_dimension_se': 'Standard error of the "coastline approximation" of the tumor',
    'radius_worst': 'Worst radius (the largest distance from the center to a perimeter point among all the tumors in the dataset)',
    'texture_worst': 'Worst texture (the standard deviation of gray-scale values among all the tumors in the dataset)',
    'perimeter_worst': 'Worst perimeter (the largest size of the core tumor among all the tumors in the dataset)',
    'area_worst': 'Worst area (the largest area of the tumor among all the tumors in the dataset)',
    'smoothness_worst': 'Worst smoothness (the smallest mean of local variation in radius lengths among all the tumors in the dataset)',
    'compactness_worst': 'Worst compactness (the largest mean of perimeter^2 / area - 1.0 among all the tumors in the dataset)',
    'concavity_worst': 'Worst concavity (the largest severity of concave portions of the contour among all the tumors in the dataset)',
    'concave points_worst': 'Worst number of concave portions of the contour among all the tumors in the dataset)',
    'symmetry_worst': 'Worst symmetry (the largest symmetry value among all the tumors in the dataset)',
    'fractal_dimension_worst': 'Worst "coastline approximation" (the largest mean of the fractal dimension of the tumor boundary among all the tumors in the dataset)',
}

# Print the attribute names and descriptions
for col in df.columns[:31]:
    if col == 'id':
        continue
    print(f'{col}: {descriptions[col]}')

#stage2
##Preprocessing

# Drop the "Unnamed: 32" column
df = df.drop(columns=['Unnamed: 32',"fractal_dimension_mean","id"])
df = df.drop(df.iloc[:, 11:31],axis = 1)


# Print data information and description
print("Data Info:")
print(df.info())

print("Data Description:")
print(df.describe())


print("\nData values and counts:")
for col in df.columns:
    print(df[col].value_counts())

print("\nNulls and other characters:")
print(df.isnull().sum())

# Separate the target variable from the features

# Map the diagnosis column to binary labels
df['diagnosis'] = (df['diagnosis'] == 'M').astype(int)

# Separate the target variable from the features
y = df['diagnosis']
X = df.iloc[:, 2:]
print("1xx")
print(X.columns)
print("1yy")
print(y)


# Normalize the data using StandardScaler
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Handle zero and NaN standard deviations
zero_std_cols = np.where(scaler.scale_ == 0)[0]
X[:, zero_std_cols] = np.nan_to_num(X[:, zero_std_cols])


print("\nNormalized data:")


##Models
# separate the target column from the input features
X = df.drop(['diagnosis',"radius_se"], axis=1)
y = df['diagnosis']
X = X.drop(columns=["symmetry_mean","smoothness_mean","texture_mean","compactness_mean"])



print("22xx")
print(X.columns)
print("22yy")
print(y)
# set the random state for reproducibility
random_state = 42

# split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=random_state)

# instantiate a logistic regression model
logreg = LogisticRegression(random_state=random_state)


# set the end date
end_date = '2023-05-03'


# perform cross validation
cv_results = cross_val_score(logreg, X, y, cv=10)

# print the mean and standard deviation of the cross validation results
print('Cross validation results:\n')
print('Mean:', cv_results.mean())
print('Standard deviation:', cv_results.std())



# fit the logistic regression model on the training data
logreg.fit(X_train, y_train)

# make predictions on the test set
y_pred = logreg.predict(X_test)
print("predictionnnn")
print(y_pred)
# print the accuracy score of the test set predictions
print('\nTest accuracy score:', accuracy_score(y_test, y_pred))

# set the end date
end_date = '2023-05-03'

# train the logistic regression model on the entire dataset
logreg.fit(X, y)

# save the trained model
joblib.dump(logreg, 'logreg_model.pkl')






##metrics
# load the saved logistic regression model
logreg = joblib.load('logreg_model.pkl')

# make predictions on the test set
y_pred = logreg.predict(X_test)

# calculate precision, recall, and F1-score
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1score = f1_score(y_test, y_pred)

# calculate confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
tn, fp, fn, tp = conf_matrix.ravel()

# calculate ROC curve and area under curve (AUC)
fpr, tpr, thresholds = roc_curve(y_test, y_pred)
roc_auc = auc(fpr, tpr)

# print the evaluation metrics
print('Precision:', precision)
print('Recall:', recall)
print('F1-Score:', f1score)
print('Confusion Matrix:')
print(conf_matrix)
print('ROC AUC:', roc_auc)



#Stage3
import mlflow
import mlflow.sklearn
import matplotlib.pyplot as plt

# Set the tracking URI to the address of your remote MLflow server
mlflow.set_tracking_uri("http://0.0.0.0:8000")
# Start an MLflow run
with mlflow.start_run():
    run = mlflow.active_run()
    run_id = run.info.run_id
    print("Run ID:", run_id)



    # Your model training code here
    model = logreg

    # Log metrics
    mlflow.log_metric("recall", recall)
    mlflow.log_metric("Precision", precision)
    mlflow.log_metric("F1-Score", f1score)
    mlflow.log_metric("ROC AUC", roc_auc)
    mlflow.log_metric("Test accuracy score", accuracy_score(y_test, y_pred))

    # Generate model signature
    input_signature = mlflow.models.signature.infer_signature(X_train)
    output_signature = mlflow.models.signature.infer_signature(model.predict(X_train))

    # Log model with signature
    mlflow.sklearn.log_model(model, "model")
    mlflow.sklearn.log_model(model, "model", signature=input_signature)
    mlflow.sklearn.log_model(model, "model", signature=output_signature)

    # Generate the plot
    plt.plot(range(len(y_pred)), y_pred)

    # Save the plot as an artifact
    plt.savefig("plot.png")

    # Log the artifact
    mlflow.log_artifact("plot.png")

    # Get the tracking URL
    tracking_url = mlflow.get_tracking_uri()

    # Print the tracking URL
    print("MLflow tracking URL:", tracking_url)


    # Load the model from the MLflow registry
    model_uri = f"runs:/{run_id}/model"
    print("Model URI:", model_uri)
    model = mlflow.sklearn.load_model(model_uri)

    # Perform inference on new data

    new_data = X_train
    predictions = model.predict(new_data)

    # Convert predictions to a DataFrame
    predictions_df = pd.DataFrame(predictions, columns=['prediction'])

    # Save the predictions to a CSV file
    predictions_df.to_csv("predictions.csv", index=False)

# End the MLflow run
mlflow.end_run()


# Get the latest run ID
latest_run = mlflow.search_runs(experiment_ids="0").iloc[0]
run_id = latest_run.run_id

# Get the model URI
model_uri = f"runs:/{run_id}/model"

# Perform MLOps tasks
mlflow.sklearn.load_model(model_uri)
# ...

# Get the tracking URL (localhost)
tracking_url = mlflow.get_tracking_uri()

# Print the tracking URL
print("MLflow tracking URL: ", tracking_url)