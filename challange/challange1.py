import joblib

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
