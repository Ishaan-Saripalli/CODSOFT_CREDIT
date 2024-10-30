import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

train_df = pd.read_csv(r'C:\\Users\\DELL\\Desktop\\CREDIT CARD\\fraudTrain.csv')
test_df = pd.read_csv(r'C:\\Users\\DELL\\Desktop\\CREDIT CARD\\fraudTest.csv')

if train_df.isnull().sum().any():
    print("Found NaN values in training data, filling them with zeros.")
    train_df.fillna(0, inplace=True)

train_df = train_df.apply(pd.to_numeric, errors='coerce')

if train_df.isnull().sum().any():
    print("Found NaN values in training data after conversion, filling them with zeros.")
    train_df.fillna(0, inplace=True)


X = train_df.drop('is_fraud', axis=1)  
y = train_df['is_fraud']              

X_train_split, X_val, y_train_split, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

sample_size = 100000  
X_train_sample = X_train_split.sample(n=sample_size, random_state=1)
y_train_sample = y_train_split.loc[X_train_sample.index]

scaler = StandardScaler()
X_train_sample = scaler.fit_transform(X_train_sample)

X_val = scaler.transform(X_val)

model = LogisticRegression(max_iter=1000)  
try:
    model.fit(X_train_sample, y_train_sample)
except ValueError as e:
    print(f"Error fitting the model: {e}")
    print("NaN values in training data:", pd.isnull(X_train_sample).sum().sum())

validation_accuracy = model.score(X_val, y_val)
print(f'Validation Accuracy: {validation_accuracy:.4f}')

if test_df.isnull().sum().any():
    print("Found NaN values in test data, filling them with zeros.")
    test_df.fillna(0, inplace=True)

test_df = test_df.apply(pd.to_numeric, errors='coerce')

if test_df.isnull().sum().any():
    print("Found NaN values in test data after conversion, filling them with zeros.")
    test_df.fillna(0, inplace=True)

X_test = test_df.drop('is_fraud', axis=1)
y_test = test_df['is_fraud']

X_test = scaler.transform(X_test)

test_accuracy = model.score(X_test, y_test)
print(f'Test Accuracy: {test_accuracy:.4f}')
