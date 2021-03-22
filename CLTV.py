# Libraries
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
pd.set_option("display.max_columns", None)

# Read data set
df_ = pd.read_excel("dataset/online_retail_II.xlsx",
                    sheet_name="Year 2010-2011")

df = df_.copy()

# General formula:
# # CLTV = (Customer_Value / Churn_Rate) x Profit_margin
# Customer_Value = Average_Order_Value * Purchase_Frequency
# Average_Order_Value = Total_Revenue / Total_Number_of_Orders
# Purchase_Frequency =  Total_Number_of_Orders / Total_Number_of_Customers
# Churn_Rate = 1 - Repeat_Rate
# Profit_margin

# CLTV: It is the monetary value that a customer will bring to this company
# in the process of relationship-communication with a company.

# Size information
df.shape

# Total number of missing observations for each variable
df.isnull().sum()

# Descriptive statistics for quantitative variables
df.describe().T


# How many values ​​are available with a unique invoice number and customer number
df["Invoice"].nunique()
df["Customer ID"].nunique()

# Data Preparation
df = df[~df["Invoice"].str.contains("C", na=False)]
df = df[(df['Quantity'] > 0)]
df.dropna(inplace=True)
df["TotalPrice"] = df["Quantity"] * df["Price"]


cltv_df = df.groupby('Customer ID').agg({'Invoice': lambda x: len(x),
                                         'Quantity': lambda x: x.sum(),
                                         'TotalPrice': lambda x: x.sum()})


# Let's change the names of the variables
cltv_df.columns = ['total_transaction', 'total_unit', 'total_price']

# Average_Order_Value = Total_Revenue / Total_Number_of_Orders
cltv_df["avg_order_value"] = cltv_df["total_price"]/cltv_df["total_transaction"]

# Purchase_Frequency = Total Number of Orders/Total Number of Customers
cltv_df["purchase_frequency"] = cltv_df['total_transaction'] / cltv_df.shape[0]

# Churn_Rate = 1 - Repeat_Rate
repeat_rate = cltv_df[cltv_df.total_transaction > 1].shape[0] / cltv_df.shape[0]
churn_rate = 1 - repeat_rate


# Calculate Profit Margin
cltv_df['profit_margin'] = cltv_df['total_price'] * 0.05

# Calculate Customer Lifetime Value
cltv_df['CV'] = (cltv_df['avg_order_value'] * cltv_df["purchase_frequency"]) / churn_rate
cltv_df['CLTV'] = cltv_df['CV'] * cltv_df['profit_margin']


# CLTV standardization process
scaler = MinMaxScaler(feature_range=(1, 100))
scaler.fit(cltv_df[["CLTV"]])
cltv_df["SCALED_CLTV"] = scaler.transform(cltv_df[["CLTV"]])

# Segmentation process
cltv_df["segment"] = pd.qcut(cltv_df["SCALED_CLTV"], 4, labels=["D", "C", "B", "A"])


# Calculation of count, mean and sum values for each variable in segment breakdowns
cltv_df.groupby("segment")[["total_transaction", "total_unit", "total_price", "CLTV", "SCALED_CLTV"]].agg(
    {"count", "mean", "sum"})

