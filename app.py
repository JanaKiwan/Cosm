import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO

st.title("Customer Data Analytics Pipeline")

uploaded_file = st.file_uploader("Upload your raw transactional data (Excel)", type=["xlsx", "xls"])
if uploaded_file:
    header_row = st.number_input("Enter the row number for headers (1-based index, default: 1)", min_value=1, value=1) - 1

    try:
        raw_data = pd.read_excel(uploaded_file, header=header_row)
        raw_data.columns = raw_data.columns.str.strip()  # Normalize column names
        st.write("### Uploaded Raw Data:")
        st.dataframe(raw_data.head())

        st.write("### Column Names in Uploaded Data:")
        st.write(list(raw_data.columns))

        def clean_data(df):
            if 'COUNTRYNAME' not in df.columns:
                st.error("Error: The column 'COUNTRYNAME' is missing.")
                return df
            df['COUNTRYNAME'] = df['COUNTRYNAME'].replace({
                'UNITED KINGDOM': 'United Kingdom',
                'BAHRAIN': 'Bahrain',
                'IVORY COAST': "Côte d'Ivoire"
            }).str.title()
            df = df[~((df['AMOUNT'] < 0) & (df['VOLUME'] >= 0))]
            df = df[~((df['VOLUME'] < 0) & (df['AMOUNT'] >= 0))]
            return df

        cleaned_data = clean_data(raw_data)
        st.write("### Cleaned Data:")
        st.dataframe(cleaned_data.head())

        def engineer_features(df):
            required_columns = ['YEAR', 'PERIOD', 'CUSTOMERNAME', 'AMOUNT']
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                st.error(f"Missing required columns: {', '.join(missing_columns)}")
                return df
            df['Date'] = pd.to_datetime(df['YEAR'].astype(str) + '-' + df['PERIOD'].astype(str) + '-01', errors='coerce')
            df['Customer_Transactions'] = df.groupby('CUSTOMERNAME')['AMOUNT'].transform('count')
            df['First_Purchase'] = df.groupby('CUSTOMERNAME')['Date'].transform('min')
            df['Last_Purchase'] = df.groupby('CUSTOMERNAME')['Date'].transform('max')
            df['Customer_Lifetime'] = (df['Last_Purchase'] - df['First_Purchase']).dt.days // 30
            df['Months_Since_Last_Purchase'] = (df['Date'].max() - df['Last_Purchase']).dt.days // 30
            df['Is_Repeat_Customer'] = df['Customer_Transactions'] > 1
            return df

        engineered_data = engineer_features(cleaned_data)
        st.write("### Feature-Engineered Data:")
        st.dataframe(engineered_data.head())

        def aggregate_features_by_customer(df):
            if 'CUSTOMERNAME' not in df.columns:
                st.error("Error: 'CUSTOMERNAME' column is required for aggregation.")
                return df
            agg_dict = {
                'AMOUNT': [('Total_Amount_Purchased', lambda x: x[x > 0].sum()),
                           ('Max_Amount_Purchased', lambda x: x[x > 0].max()),
                           ('Min_Amount_Purchased', lambda x: x[x > 0].min())],
                'VOLUME': [('Total_Volume_Purchased', lambda x: x[x > 0].sum())],
                'ITEMGROUPDESCRIPTION': [('Most_Frequent_Item_Group', lambda x: x.mode().iloc[0] if not x.mode().empty else None)]
            }
            agg_df = df.groupby('CUSTOMERNAME').agg(
                **{new_col: (col, func) for col, entries in agg_dict.items() for new_col, func in entries}
            ).reset_index()
            return agg_df

        aggregated_data = aggregate_features_by_customer(engineered_data)
        st.write("### Aggregated Customer Data:")
        st.dataframe(aggregated_data.head())

        def convert_to_excel(df):
            output = BytesIO()
            with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                df.to_excel(writer, index=False, sheet_name='Aggregated Data')
            return output.getvalue()

        excel_data = convert_to_excel(aggregated_data)
        st.download_button("Download Aggregated Data as Excel", data=excel_data, file_name="aggregated_data.xlsx")
    except Exception as e:
        st.error(f"An error occurred: {e}")
