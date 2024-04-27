#!/usr/bin/env python
# coding: utf-8

# # Customer Segment Analysis_RFM
# Dataset from dacon 
# 
# * R(Recency)
# * F(Frequency)
# * M(Monetary)

# Segment
# VIP (0.1%)
# Loyal (1.7%)
# Potential Loyal (14.7%)
# Must-Not-Lose (2.9%)
# Recent New (30.7%)
# At-Risk (16.8%)
# Others

# **Table of contents**
# 1. Data Preprocessing
# 1. Data Analysis & Visualizations
# 1. Modeling

# In[1]:


# import library
import pandas as pd
import numpy as np
import datetime as dt

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')

from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules

import warnings
warnings.filterwarnings('ignore')


# In[2]:


# Data load
customer = pd.read_csv('/kaggle/input/customer-segment/Customer_info.csv')
discount = pd.read_csv('/kaggle/input/customer-segment/Discount_info.csv')
marketing = pd.read_csv('/kaggle/input/customer-segment/Marketing_info.csv')
onlinesales = pd.read_csv('/kaggle/input/customer-segment/Onlinesales_info.csv')
tax = pd.read_csv('/kaggle/input/customer-segment/Tax_info.csv')


# In[3]:


# Initial Dataset VIEW!
print(f"<Customer data>\n{customer.isnull().sum()}\n")
print(f"<Discount data>\n{discount.isnull().sum()}\n")
print(f"<Marketing data>\n{marketing.isnull().sum()}\n")
print(f"<Onlinesales data>\n{onlinesales.isnull().sum()}\n")
print(f"<Tax data>\n{tax.isnull().sum()}\n")


# In[4]:


# Data shape
print(customer.shape)
print(discount.shape)
print(marketing.shape)
print(onlinesales.shape)
print(tax.shape)


# In[5]:


# Translate korean columns to English
customer.columns = ["Customer_ID", "Sex", "Location", "Membership Duration"]
discount.columns = ["Month", "Category", "Coupon_code", "Discount(%)"]
marketing.columns = ["Date", "Offline", "Online"]
onlinesales.columns = ["Customer_ID", "Transaction_ID", "Transaction_date", "Product_ID","Category", "Qty", "Average_price", "Delivery_fee","Coupon_status"]
tax.columns = ["Category", "GST"]


# In[6]:


# Let's check the customer's location
customer.Location.unique()


# In[7]:


# Map visualizations by Customer location
$ pip install folium
import folium

location_data = {
    'Chicago': {'lat': 41.8781, 'long': -87.6298},
    'California': {'lat': 36.7783, 'long': -119.4179},
    'New York': {'lat': 40.7128, 'long': -74.0060},
    'New Jersey': {'lat': 40.0583, 'long': -74.4057},
    'Washington DC': {'lat': 38.9072, 'long': -77.0369}
}

map = folium.Map(location=[39.8283, -98.5795], zoom_start=4)

for city, coord in location_data.items():
    folium.Marker(
        location=[coord['lat'], coord['long']],
        popup=f'{city}',
        icon=folium.Icon(color='red', icon='info-sign')
    ).add_to(map)

map


# **Data Preprocessing**

# In[8]:


# Month data change to numeric 
# Month mapping
month_mapping = {'Jan': 1, 'Feb': 2, 'Mar': 3, 'Apr': 4, 'May': 5, 'Jun': 6,
                 'Jul': 7, 'Aug': 8, 'Sep': 9, 'Oct': 10, 'Nov': 11, 'Dec': 12}
discount["Month"] = discount["Month"].map(month_mapping)
onlinesales["Transaction_date"] = pd.to_datetime(onlinesales['Transaction_date'])
onlinesales['Month'] = onlinesales['Transaction_date'].dt.month

# online, customer join on Customer_ID 
df = pd.merge(onlinesales, customer, on='Customer_ID', how='left')

# df, discount join on Month and Product_category
df = pd.merge(df, discount, on=['Month', 'Category'], how='left')

# df, tax join on Product_category
df = pd.merge(df, tax, on='Category', how='left')


# In[9]:


# check for missing values again
df.isnull().sum()


# I have observed missing values in the 'Coupon_code' and 'Discount(%)' columns.

# In[10]:


# Fill in the missing values in the 'Coupon_code' column with 'unknown' and the missing values in the 'Discount(%)' column with 0.
df['Coupon_code'].fillna("Unknown", inplace = True)
df["Discount(%)"].fillna(0, inplace = True)


# **Calculate RFM metrics and classify customer segments.**
# 
# (1) Calculate Recency, Frequency, and Monetary metrics.

# In[11]:


df.head()


# In[12]:


# Create "Total" column 
df["Total"] = df["Qty"] * df["Average_price"]

# Define a function to calculate the total amount paid.
def total(row):
    price = row["Total"]
    gst = row["GST"]
    discount_rate = row["Discount(%)"] if row ["Coupon_status"] == "Used" else 0
    subtotal = price*(1-discount_rate/100)
    total = subtotal + (subtotal*gst)
    return total

df["Paid_amount"] = df.apply(total, axis=1)
# Excluding shipping fees


# In[13]:


df.head()


# To determine the total amount paid by a customer in a single transaction, a 'Total' column was created by multiplying 'Qty' and 'Average_Price'. Additionally, a 'Paid_amount' column was created to reflect the actual amount paid in the transaction, taking into account the discount rate for customers who used coupons and applying taxes by product category.

# In[14]:


# Let's check the shipping fees by transaction ID.
df.groupby("Transaction_ID")["Delivery_fee"].nunique().value_counts()
#If the transaction IDs are the same, then the shipping fees are the same.


# In[15]:


# Find the first shipping fee for each customer ID and transaction ID = unique shipping fee per customer ID and transaction ID.
first_delivery_fee_by_customer = df.groupby(['Customer_ID', "Transaction_ID"]).first()["Delivery_fee"]

# Match and store the sum of the shipping fees with each customer ID.
customer_delivery_fee_sum = first_delivery_fee_by_customer.groupby("Customer_ID").sum()


# In[16]:


first_delivery_fee_by_customer.head()


# In[17]:


customer_delivery_fee_sum.head()


# **🎈LET'S SET THE RFM!!**

# In[18]:


# Set the day after the last day of the data as the reference point.
last = df['Transaction_date'].max() + pd.DateOffset(days=1)
print(last)


# The latest day is 2020-01-01 00:00:00

# In[19]:


rfm_df = df.groupby(["Customer_ID"]).agg({
    "Transaction_date" : lambda x : (last - x.max()).days,
    "Transaction_ID" : lambda x : x.nunique(),
    "Paid_amount" : "sum"})

rfm_df.rename(columns = {"Transaction_date" : "Recency",
                        "Transaction_ID" : "Frequency",
                        "Paid_amount" : "Monetary"}, inplace = True)
rfm_df.reset_index(inplace=True)
rfm_df.head()


# In[20]:


# Sum Delivery fee
for customer_id, delivery_fee in customer_delivery_fee_sum.items():
    rfm_df.loc[rfm_df["Customer_ID"] == customer_id, "Monetary"] += delivery_fee
rfm_df.head()


# In[21]:


# merge data on "df"
df = df.merge(rfm_df, on = "Customer_ID")


# 
# I believe it is appropriate to differentiate the R (Recency) or F (Frequency) metrics by product category. For example, purchasing a laptop every three months might be considered similar in value to purchasing clothing every week.
# 
# To implement this, I assigned weights based on product categories. The weights were calculated by determining the time difference between the first and second purchase dates for each category, averaged across customers. This average was then added to the transaction date to adjust the Recency metric accordingly.
# 

# In[22]:


# Proceed only for the top 10 categories.
df["Category"].value_counts().head(10)


# In[23]:


df.head()


# In[24]:


categories = { 'Office','Apparel','Nest-USA','Drinkware','Lifestyle',
    'Nest','Bags','Headgear','Notebooks & Journals','Waze'}

def calculate_average_difference(df, category):
    category_transactions = df[df["Category"] == category]
    category_transactions = category_transactions.sort_values(by = ["Customer_ID", "Transaction_date"])
    customer_groups = category_transactions.groupby("Customer_ID")["Transaction_date"]
    differences = []
    for customer_id, dates in customer_groups:
        date_list = dates.tolist()
        if len(date_list) >= 2:
            difference = date_list[1] - date_list[0]
            differences.append(difference)
    if differences : 
        average_difference = sum(differences, pd.Timedelta(0)) / len(differences)
        return average_difference
    else:
        return None
for category in categories:
    average_difference = calculate_average_difference(df, category)
    if average_difference is not None:
        print(f"'{category}':", average_difference)
    else:
        print(f"There is no data of '{category}'")


# In[25]:


# Define the values corresponding to each product category.
product_category_values = {'Office': 9,'Apparel': 6,'Nest-USA': 5,
    'Drinkware': 13,'Lifestyle': 17,'Nest': 4,'Bags': 18,
    'Headgear': 27,'Notebooks & Journals': 20,'Waze': 23}

# Add values to the transaction dates for each product category.
for category, value in product_category_values.items():
    df.loc[df["Category"] == category, "Transaction_date"] += pd.Timedelta(days = value)
df["Transaction_date"] = pd.to_datetime(df["Transaction_date"])
last = df["Transaction_date"].max() + pd.DateOffset(days = 27)
a = pd.DataFrame()
a = df.groupby(["Customer_ID"]).agg({"Transaction_date" : lambda x : (last - x.max()).days})
a.rename(columns = {"Transaction_date" : "Recency"}, inplace=True)
a.reset_index(inplace=True)
df["Recency"] = df["Customer_ID"].map(a.set_index("Customer_ID")["Recency"])


# In[26]:


a.head()


# **Segment R (Recency), F (Frequency), M (Monetary)**
# 1) Recency

# In[27]:


# Visualize the distribution of Recency for all customers.
plt.figure(figsize=(10, 6))
plt.hist(df['Recency'], bins=40, color='skyblue', edgecolor='black')
plt.title('Distribution of Recency')
plt.xlabel('Recency')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()


# In[28]:


print(df["Recency"].min())
print(df["Recency"].max())


# The Recency of all customers ranges from 27 to 411, and it displays the distribution mentioned above.
