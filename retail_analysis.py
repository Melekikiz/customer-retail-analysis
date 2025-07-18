import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder

file_path="/Users/melekikiz/MachineLeraning/Customer Satisfaction/EDA/Online Retail.xlsx"

df = pd.read_excel(file_path)

#Handling missing descriptions
print("Number of rows with missing 'Description':", df['Description'].isnull().sum())
df_clean=df.dropna(subset=['Description']).copy()
print("Lenght of cleaned data:" , len(df_clean))

#Convert date and customer ID
if 'InvoiceDate' in df.columns:
    df['InvoiceDate']=pd.to_datetime(df['InvoiceDate'], errors='coerce')

df_clean['CustomerID']=df_clean['CustomerID'].astype('str')

#Basic info
print("First 5 rows:")
print(df.head())

print("Data info:")
print(df.info())

print("Missing values:")
print(df.isnull().sum())

#Column classification
categorical_cols=df.select_dtypes(include=['object']).columns.tolist()
numerical_cols=df.select_dtypes(include=['int64', 'float64']).columns.tolist()
datetime_cols=df.select_dtypes(include=['datetime64[ns]']).columns.tolist()

print("Categorical columns:", categorical_cols)
print("Numerical columns:", numerical_cols)
print("Datetime columns:", datetime_cols)

#Unique country and product count
print("Unique Country count:", df_clean['Country'].nunique())
print("Unique Product count:", df_clean['Description'].nunique())

#Top products
top_products=df_clean['Description'].value_counts().head(10)
print("En çok satan 10 ürün:")
print(top_products)

#Top countries
top_countries= df['Country'].value_counts().head(5)

le_country=LabelEncoder()
df_clean['CountryEncoded']=le_country.fit_transform(df_clean['Country'])

print("\nEncoded country column (first 5 rows):")
print(df_clean[['Country', 'CountryEncoded']].head())

#Top customers
top_customers=df_clean['CustomerID'].value_counts().head(10)
print("Top 10 customers by number of orders (CustomerID):")
print(top_customers)

#Add year and month
df_clean['Year']=df_clean['InvoiceDate'].dt.year
df_clean['Month']=df_clean['InvoiceDate'].dt.month

#Grouped sales data
sales_by_year=df_clean.groupby('Year')['Quantity'].sum()
sales_by_month=df_clean.groupby('Month')['Quantity'].sum()




#Visualization: Top 5 countries
plt.figure(figsize=(8, 5))
sns.barplot(x=top_countries.values, y=top_countries.index, hue=None, palette='viridis', legend=False)
plt.title('Top 5 Countries by Order Count')
plt.xlabel('Number of Orders')
plt.ylabel('Country')
plt.show()

#Visualization: Top 10 products
plt.figure(figsize=(12, 7))
sns.barplot(x=top_products.values, y=top_products.index,  palette='magma')
plt.title('Top 10 Best-Selling Products')
plt.xlabel('Quantity Sold')
plt.ylabel('Product')
plt.show()

#Histograms: Quantity and Unit Price
plt.figure(figsize=(10,5))

plt.subplot(1, 2, 1)
sns.histplot(df_clean['Quantity'], bins=50, kde=False)
plt.title('Distribution of Quantity')
plt.xlabel('Quantity')

plt.subplot(1,2,2)
sns.histplot(df_clean['UnitPrice'], bins=50, kde=False, color='orange')
plt.title('Distribution of Unit Price')
plt.xlabel('Unit Price')

plt.tight_layout()
plt.show()


#Sales by year and month
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
sales_by_year.plot(kind='bar', color='green')
plt.title('Yearly Sales Quantity')
plt.xlabel('Year')
plt.ylabel('Quantity Sold')

plt.subplot(1,2,2)
sales_by_month.plot(kind='bar', color='purple')
plt.title('Monthly Sales Quantity')
plt.xlabel('Month')
plt.ylabel('Quantity Sold')

plt.tight_layout()
plt.show()