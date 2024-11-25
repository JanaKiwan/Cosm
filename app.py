import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO  # For creating Excel in-memory files

# Title
st.title("Customer Data Analytics Pipeline")

# File Upload Section
uploaded_file = st.file_uploader("Upload your raw transactional data (Excel)", type=["xlsx", "xls"])
if uploaded_file is not None:
    # Load data
    raw_data = pd.read_excel(uploaded_file)
    st.write("### Uploaded Raw Data:")
    st.dataframe(raw_data.head())

    # Data Cleaning Function
    def clean_data(df):
        # Country name corrections
        country_name_corrections = {
            'UNITED KINGDOM': 'United Kingdom',
            'BAHRAIN': 'Bahrain',
            'IVORY COAST': "Côte d'Ivoire"
        }
        df.loc[:, 'COUNTRYNAME'] = df['COUNTRYNAME'].replace(country_name_corrections).str.title()

        # Remove inconsistent records
        df = df[~((df['AMOUNT'] < 0) & (df['VOLUME'] >= 0))]
        df = df[~((df['VOLUME'] < 0) & (df['AMOUNT'] >= 0))]
        return df

    cleaned_data = clean_data(raw_data)
    st.write("### Cleaned Data:")
    st.dataframe(cleaned_data.head())

    # Feature Engineering
    def engineer_features(df):
        df['Date'] = pd.to_datetime(df['YEAR'].astype(str) + '-' + df['PERIOD'].astype(str) + '-01')
        df['Customer_Transactions'] = df.groupby('CUSTOMERNAME')['AMOUNT'].transform('count')
        df['First_Purchase'] = df.groupby('CUSTOMERNAME')['Date'].transform('min')
        df['Last_Purchase'] = df.groupby('CUSTOMERNAME')['Date'].transform('max')
        most_recent_date = df['Date'].max()
        df['Customer_Lifetime'] = (df['Last_Purchase'] - df['First_Purchase']).dt.days // 30
        df['Months_Since_Last_Purchase'] = (most_recent_date - df['Last_Purchase']).dt.days // 30
        df['Is_Repeat_Customer'] = df['Customer_Transactions'] > 1
        return df

    engineered_data = engineer_features(cleaned_data)
    st.write("### Feature-Engineered Data:")
    st.dataframe(engineered_data.head())

    # Aggregation Function
    def aggregate_features_by_customer(df):
        agg_dict = {
            'AMOUNT': [('Total_Amount_Purchased', lambda x: x[x > 0].sum()),
                       ('Max_Amount_Purchased', lambda x: x[x > 0].max()),
                       ('Min_Amount_Purchased', lambda x: x[x > 0].min())],
            'VOLUME': [('Total_Volume_Purchased', lambda x: x[x > 0].sum()),
                       ('Max_Volume_Purchased', lambda x: x[x > 0].max()),
                       ('Min_Volume_Purchased', lambda x: x[x > 0].min())],
            'ITEMGROUPDESCRIPTION': [('Most_Frequent_Item_Group', lambda x: x.mode().iloc[0] if not x.mode().empty else None)],
            'Customer_Transactions': [('Customer_Transactions', 'first')],
            'Months_Since_Last_Purchase': [('Months_Since_Last_Purchase', 'first')],
            'Customer_Lifetime': [('Customer_Lifetime', 'first')]
        }

        # Flatten the aggregation
        agg_df = df.groupby('CUSTOMERNAME').agg(
            **{new_col: (col, func) for col, entries in agg_dict.items() for new_col, func in entries}
        ).reset_index()
        return agg_df

    aggregated_data = aggregate_features_by_customer(engineered_data)
    st.write("### Aggregated Customer Data:")
    st.dataframe(aggregated_data.head())

    # Refund and Purchase Metrics Calculation
    def calculate_refund_and_purchase_metrics(df):
        df['Refund_Ratio'] = df['Total_Amount_Purchased'].abs() / (
            df['Total_Amount_Purchased'] + df['Total_Amount_Purchased'].abs())
        df['Average_Purchase_Per_Month'] = df['Total_Amount_Purchased'] / df['Customer_Lifetime']
        return df

    aggregated_data = calculate_refund_and_purchase_metrics(aggregated_data)
    st.write("### Aggregated Data with Metrics:")
    st.dataframe(aggregated_data.head())

    # Data Download as Excel
    @st.cache_data
    def convert_to_excel(df):
        output = BytesIO()
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            df.to_excel(writer, index=False, sheet_name='Aggregated Data')
        processed_data = output.getvalue()
        return processed_data

    excel_data = convert_to_excel(aggregated_data)
    st.download_button(
        label="Download Aggregated Data as Excel",
        data=excel_data,
        file_name='aggregated_customer_data.xlsx',
        mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
    )

    # Visualizations
    st.write("### Refund Trends Over Time")
    refunds_over_time = engineered_data.groupby('Date')['AMOUNT'].sum().reset_index()
    refunds_over_time['AMOUNT'] = refunds_over_time['AMOUNT'].abs()

    fig, ax = plt.subplots()
    ax.plot(refunds_over_time['Date'], refunds_over_time['AMOUNT'], marker='o')
    ax.set_title('Refund Trends Over Time')
    ax.set_xlabel('Date')
    ax.set_ylabel('Refund Amount')
    st.pyplot(fig)

    st.write("### Customer Lifetime Distribution")
    fig, ax = plt.subplots()
    ax.hist(engineered_data['Customer_Lifetime'], bins=20, color='skyblue')
    ax.set_title('Customer Lifetime Distribution')
    ax.set_xlabel('Months')
    ax.set_ylabel('Frequency')
    st.pyplot(fig)
