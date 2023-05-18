import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import seaborn as sns
import matplotlib.pyplot as plt




#load data
df = pd.read_csv('/Users/mertatalay/Desktop/programacionII/challange/data.csv')

#drop unnamed
df = df.drop(columns=['Unnamed: 32',"id","fractal_dimension_mean"])
df = df.drop(df.iloc[:, 10:31],axis = 1)
df = df.drop(columns=["symmetry_mean","smoothness_mean","texture_mean"])

print(df.columns)
#set target in binary labels

df['diagnosis'] = (df['diagnosis'] == 'M').astype(int)
# Separate the target variable from the features
y = df['diagnosis']
X = df.iloc[:, 2:]

print(df[df.columns[-1]])
# Create a correlation matrix
correlation_matrix = df.corr()

# Visualize the correlation matrix using a heatmap
plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Correlation Matrix')
plt.show()

# Display the correlation matrix
print(correlation_matrix)
plt.plot(correlation_matrix)
plt.savefig("plot1.png")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
# Using statsmodels
logit_model = sm.Logit(y_train, X_train)
result = logit_model.fit()

# Using scikit-learn
logreg = LogisticRegression()
logreg.fit(X_train, y_train)


# Using statsmodels
print(result.summary())


# Using scikit-learn
y_pred = logreg.predict(X_test)

print(y_pred)
