from sklearn.model_selection import train_test_split
import matplotlib as matplotlib
import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
import keras

df = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")
# Since customer ID is not required we drop it and do sample exploration
df.drop('customerID', axis='columns', inplace=True)
print(df.dtypes)


print(df.shape)
print(df[df.TotalCharges == ' '])
# We need to remove the Total charges entries before converting type object to float64 bits
df1 = df[df.TotalCharges != ' '].copy()
print(df1.shape)
# Clearly note that there are total of 11 entries/rows which have no NULL value in the TotalCharges COL.
df1.TotalCharges = pd.to_numeric(df1.TotalCharges)


# Now we create the bar graph on the basis of the tenure of the clients based on churning
# and non churning.

# tenure_churn_no = df1[df1.Churn == 'No'].tenure
# tenure_churn_yes = df1[df1.Churn == 'Yes'].tenure

# plt.hist([tenure_churn_yes, tenure_churn_no], color=[
#          'red', 'green'], label=['Churn=Yes', 'Churn=No'])
# plt.legend()
# plt.xlabel('Tenure (months)')  # Label for x-axis
# plt.ylabel('Number of Customers')  # Label for y-axis
# plt.show()

# After plotting the histogram , we clearly observe that people who are with the company for longer time DON'T CHURN.
# We further do the cleaning of the data for applying Neural Network.


def print_unique_col_values(df):
    for column in df:
        if df[column].dtypes == 'object':
            print(f'{column}: {df[column].unique()}')


print_unique_col_values(df1)
# Now since no phone service/ No internet service implys directly No so we change them.

df1.replace('No internet service', 'No', inplace=True)
df1.replace('No phone service', 'No', inplace=True)

# Replace the genders Male and Female with 1 and 0
print_unique_col_values(df1)
df1['gender'].replace({'Female': 1, 'Male': 0}, inplace=True)

# Replace the True False columns with 1 and 0
yes_no_columns = [
    'Partner',
    'Dependents',
    'PhoneService',
    'MultipleLines',
    'InternetService',
    'OnlineSecurity',
    'OnlineBackup',
    'DeviceProtection',
    'TechSupport',
    'StreamingTV',
    'StreamingMovies',
    'PaperlessBilling',
    'Churn'
]

for col in yes_no_columns:
    df1[col].replace({'Yes': 1, 'No': 0}, inplace=True)

print_unique_col_values(df1)

for col in df1:
    print(f'{col}: {df1[col].unique()}')

# Now we create a dummy rows for Contract, Internet Service, PaymentMethod

df2 = pd.get_dummies(data=df1, columns=[
                     'InternetService', 'Contract', 'PaymentMethod'])
print(df2.columns)

# Now we need to scale some columns as they are not in terms of 0 or 1
cols_to_scale = ['tenure', 'MonthlyCharges', 'TotalCharges']
scaler = MinMaxScaler()
df2[cols_to_scale] = scaler.fit_transform(df2[cols_to_scale])

print(df2.sample(5))

# Now we create the file for training and testing purposes.
X = df2.drop('Churn', axis='columns')
Y = df2['Churn']

X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, random_state=5)

# print(len(X_train.columns))
# Now we create a neural network which predicts the test data.

model = keras.Sequential([
    # for hidden layer use relu function
    keras.layers.Dense(20, input_shape=(26,), activation='relu'),
    keras.layers.Dense(1, activation='sigmoid'),
])

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])
Y_train = Y_train.astype('float32')
Y_test = Y_test.astype('float32')
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
model.fit(X_train, Y_train, epochs=100)
# Now we have created the model and we will test it on out TEST data. Create the confusion matrix.
model.evaluate(X_test, Y_test)
yp = model.predict(X_test)

print(yp)
# See clearly that the evaulated values of the metrics are real numbers between 0 and 1
Y_pred = []
for element in yp :
    if element > 0.5:
        Y_pred.append(1)
    else:
        Y_pred.append(0)
        
from sklearn.metrics import confusion_matrix ,classification_report
print(classification_report(Y_test,Y_pred))

import seaborn as sn
confmat =tf.math.confusion_matrix(labels=Y_test,predictions= Y_pred)
plt.figure(figsize= (10,7))
sn.heatmap(confmat , annot = True ,fmt= 'd')
plt.xlabel('Predicted')
plt.ylabel('Truth')
plt.show()
