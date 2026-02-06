import pandas as pd

print("Script started")

df = pd.read_csv(
    r"C:\Users\bumik\OneDrive\Documents\New folder (2)\fraud-detection-system\data\raw\creditcard.csv"
)

print("Data loaded successfully")
print(df.shape)
df.head()
df.info()

df['Class'].value_counts(normalize=True) * 100
df.groupby('Class')['Amount'].describe()
