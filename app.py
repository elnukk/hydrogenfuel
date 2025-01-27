import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.preprocessing import LabelEncoder
import re
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import os
import json
import csv
from sklearn.impute import SimpleImputer
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import RMSprop
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# connecting to google drive
from google.colab import drive
drive.mount('/content/drive')
# copying file from drive to here
!cp /content/drive/MyDrive/hydrogendataset.zip /content/
# extracting file
!unzip hydrogendataset.zip -d "./data"

# list of all files in ./data
data_set = os.listdir('data')

columns = None
count = 0
with open('hydrogen.csv', 'w') as csvfile:
  for data in data_set:
    if data.split('.')[-1] == 'json':
      with open(os.path.join('data', data), 'r') as f:
        read = json.load(f)
      if count == 0:
        columns = read.keys()
        writer = csv.DictWriter(csvfile, fieldnames=columns)
        writer.writeheader()

      writer.writerow(read)
      count += 1
print(f'Written {count} jsons in the hydrogen.csv')

with open('/content/hydrogen.csv', 'r') as file:
    reader = csv.reader(file)
    headers = next(reader)
    print(headers)

from google.colab import files
files.download('/content/hydrogen.csv')

print(data.head())

# create an empty list to store the symbolElement values
symbolElements = []

# loop over all the rows in the DataFrame
for i in range(137953):
    # find the position of the first occurrence of the "symbol" string
    pos = data.loc[i,"elements"].find("symbol")

    # extract the substring starting from 2 characters after the position of the "symbol" string
    symbolElement = data.loc[i,"elements"][pos+10:pos+12].replace("'","")

    # append the symbolElement value to the list
    symbolElements.append(symbolElement)

# create a new column in the DataFrame using the list of symbolElement values
data["symbolElements"] = symbolElements
print(data.head())



# create an empty list to store the hydrogen total absorption values
hydrogenTotalAdsorb = []

for i in range(137953):
    # find the position of the first occurrence of the "symbol" string
    pos2 = data.loc[i,"isotherms"].find("total_adsorption")

    # check if the "total_adsorption" string was found
    if pos2 != -1:
        # remove any unwanted characters from the substring
        hydrogenAdsorb = data.loc[i,"isotherms"][pos2+19:pos2+26].replace("{","").replace("}","").replace("'","").replace(",","").replace("p","")
    else:
        # set the value to NaN if the string was not found
        hydrogenAdsorb = float("NaN")

    # append the symbolElement value to the list
    hydrogenTotalAdsorb.append(hydrogenAdsorb)
# create a new column in the DataFrame using the list of symbolElement values
data["hydrogenTotalAdsorb"] = hydrogenTotalAdsorb
# print(data.head())

# Convert the hydrogenTotalAdsorb list to a NumPy array
hydrogenTotalAdsorb_array = np.array(hydrogenTotalAdsorb)

# Create a new column in the DataFrame using the NumPy array
data["hydrogenTotalAdsorb"] = hydrogenTotalAdsorb_array

print(data.loc[20, "hydrogenTotalAdsorb"])



# seperate elements and group datas
sample = []
for i in range(137953):
    test = data.loc[i, 'symbolElements']
    if test == 'O':
        sample.append(data.loc[i])

sample_df = pd.DataFrame(sample)

print(sample_df.head())


# Create an instance of LabelEncoder
label_encoder = LabelEncoder()

# Encode the 'column_name' column
sample_df['encoded_symbols'] = label_encoder.fit_transform(sample_df['symbolElements'])

# Perform linear regression
X = sample_df[['void_fraction', 'surface_area_m2cm3', 'encoded_symbols']]
y = sample_df['hydrogenTotalAdsorb'].astype(float)

# Add constant to X
X = sm.add_constant(X)

# Fit the linear regression model
model = sm.OLS(y, X).fit()

# Print the regression summary
print(model.summary())
correlation_matrix = data.corr()
print(correlation_matrix)


correlation_matrix = sample_df.corr()
print(correlation_matrix)

# Perform linear regression
X = sample_df[['void_fraction', 'surface_area_m2cm3']].astype(float)
y = sample_df['hydrogenTotalAdsorb'].astype(float)

# Add constant to X
X = sm.add_constant(X)

# Fit the linear regression model
model = sm.OLS(y, X).fit()

# Print the regression summary
print(model.summary())


subset = sample_df.loc[:, ["id", "void_fraction", "surface_area_m2cm3","hydrogenTotalAdsorb"]]
subset.columns = subset.columns.str.strip()
# Create an imputer object
imputer = SimpleImputer(strategy="mean")

# Fit the imputer on the subset DataFrame
imputer.fit(subset)

# Transform the subset DataFrame by filling missing values
subset_imputed = pd.DataFrame(imputer.transform(subset), columns=subset.columns)

print(subset_imputed.head())


subset = data.loc[:, ["id", "void_fraction", "surface_area_m2cm3","hydrogenTotalAdsorb"]]
subset.columns = subset.columns.str.strip()
# Create an imputer object
imputer = SimpleImputer(strategy="mean")

# Fit the imputer on the subset DataFrame
imputer.fit(subset)

# Transform the subset DataFrame by filling missing values
subset_imputed = pd.DataFrame(imputer.transform(subset), columns=subset.columns)

print(subset_imputed.head())


X = subset_imputed.drop("id", axis=1)
y = data["hydrogenTotalAdsorb"]


# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


missing_values = y.isnull().sum()
print("Number of missing values in y:", missing_values)


print("X_train shape:", X_train.shape)
print("y_train shape:", y_train.shape)

missing_values = X_train.isnull().sum()
print("Number of missing values in X_train:\n", missing_values)


# Convert target variable to numeric
y = pd.to_numeric(y)

# Perform train-validation-test split
X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.25, random_state=42)

# Normalize the feature data
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)

# Build the model
model = Sequential()
model.add(Dense(20, input_shape=(3,), activation='relu'))
model.add(Dense(30, activation='relu'))
model.add(Dense(25, activation='relu'))
model.add(Dense(units=1))

model.compile(optimizer=RMSprop(lr=0.0015), loss='mean_squared_error')

# Train the model and record training history
history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=50, batch_size=32, verbose=0)

# Make predictions on the training, validation, and testing data
y_train_pred = model.predict(X_train)
y_val_pred = model.predict(X_val)
y_test_pred = model.predict(X_test)

# Calculate R2 score
train_score = r2_score(y_train, y_train_pred)
val_score = r2_score(y_val, y_val_pred)
test_score = r2_score(y_test, y_test_pred)

print("R2 Score for the training set is:", train_score)
print("R2 Score for the validation set is:", val_score)
print("R2 Score for the test set is:", test_score)

# Plot the learning curve
import matplotlib.pyplot as plt

train_loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(train_loss) + 1)

plt.plot(epochs, train_loss, 'b-', label='Training Loss')
plt.plot(epochs, val_loss, 'r-', label='Validation Loss')
plt.title('Learning Curve')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()


# Convert target variable to numeric
y_train = pd.to_numeric(y_train)
y_test = pd.to_numeric(y_test)

# Build and train the model
model = Sequential()
model.add(Dense(20, input_shape=(2,), activation='relu'))
model.add(Dense(30, activation='relu'))
model.add(Dense(25, activation='relu'))
model.add(Dense(units=1))

model.compile(optimizer=RMSprop(lr=0.0015),
              loss='mean_squared_error')

# Train the model
history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=50, batch_size=32)

# Make predictions on the training and testing data
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

# Calculate R2 score
train_score = r2_score(y_train, y_train_pred)
test_score = r2_score(y_test, y_test_pred)

print("R2 Score for the training set is:", train_score)
print("R2 Score for the test set is:", test_score)

