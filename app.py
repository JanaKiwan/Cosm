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
        # Load data
        raw_data = pd.read_excel(uploaded_file, header=header_row)
        raw_data.columns = raw_data.columns.str.strip()  # Normalize column names
        st.write("### Uploaded Raw Data:")
        st.dataframe(raw_data.head())

        st.write("### Column Names in Uploaded Data:")
        st.write(list(raw_data.columns))

        # Data Cleaning Function
        def clean_data(df):
            if 'COUNTRYNAME' not in df.columns:
                st.error("Error: The column 'COUNTRYNAME' is missing.")
                return df
            # Normalize country names
            df['COUNTRYNAME'] = df['COUNTRYNAME'].replace({
                'UNITED KINGDOM': 'United Kingdom',
                'BAHRAIN': 'Bahrain',
                'IVORY COAST': "Côte d'Ivoire"
            }).str.title()
            # Remove inconsistent records
            df = df[~((df['AMOUNT'] < 0) & (df['VOLUME'] >= 0))]
            df = df[~((df['VOLUME'] < 0) & (df['AMOUNT'] >= 0))]
            return df

        cleaned_data = clean_data(raw_data)
        st.write("### Cleaned Data:")
        st.dataframe(cleaned_data.head())

        # Feature Engineering
        def engineer_features(df):
            required_columns = ['YEAR', 'PERIOD', 'CUSTOMERNAME', 'AMOUNT']
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                st.error(f"Missing required columns: {', '.join(missing_columns)}")
                return df

            # Date-related features
            df['Date'] = pd.to_datetime(df['YEAR'].astype(str) + '-' + df['PERIOD'].astype(str) + '-01', errors='coerce')
            df['Customer_Transactions'] = df.groupby('CUSTOMERNAME')['AMOUNT'].transform('count')
            df['First_Purchase'] = df.groupby('CUSTOMERNAME')['Date'].transform('min')
            df['Last_Purchase'] = df.groupby('CUSTOMERNAME')['Date'].transform('max')
            df['Customer_Lifetime'] = (df['Last_Purchase'] - df['First_Purchase']).dt.days // 30
            df['Months_Since_Last_Purchase'] = (df['Date'].max() - df['Last_Purchase']).dt.days // 30

            # Time between purchases
            df['Time_Between_Purchases'] = df.groupby('CUSTOMERNAME')['Date'].diff().dt.days // 30
            df['Mean_Time_Between_Purchases'] = df.groupby('CUSTOMERNAME')['Time_Between_Purchases'].transform('mean').fillna(0)
            df['Std_Dev_Time_Between_Purchases'] = df.groupby('CUSTOMERNAME')['Time_Between_Purchases'].transform('std').fillna(0)
            df['Max_Time_Without_Purchase'] = df.groupby('CUSTOMERNAME')['Time_Between_Purchases'].transform('max').fillna(0)

            # Repeat customer flag
            df['Is_Repeat_Customer'] = df['Customer_Transactions'] > 1
            return df

        engineered_data = engineer_features(cleaned_data)
        st.write("### Feature-Engineered Data:")
        st.dataframe(engineered_data.head())

        # Calculate Seasonality Indices
        def calculate_seasonality_indices(df):
            def assign_rolling_year(row):
                if row['YEAR'] == 2020 and row['PERIOD'] >= 9:
                    return '20-21'
                elif row['YEAR'] == 2021 and row['PERIOD'] <= 8:
                    return '20-21'
                elif row['YEAR'] == 2021 and row['PERIOD'] >= 9:
                    return '21-22'
                elif row['YEAR'] == 2022 and row['PERIOD'] <= 8:
                    return '21-22'
                elif row['YEAR'] == 2022 and row['PERIOD'] >= 9:
                    return '22-23'
                elif row['YEAR'] == 2023 and row['PERIOD'] <= 8:
                    return '22-23'
                elif row['YEAR'] == 2023 and row['PERIOD'] >= 9:
                    return '23-24'
                elif row['YEAR'] == 2024 and row['PERIOD'] <= 8:
                    return '23-24'
                else:
                    return None

            df['Rolling_Year'] = df.apply(assign_rolling_year, axis=1)

            rolling_year_transactions = (
                df.groupby(['CUSTOMERNAME', 'Rolling_Year'])['CUSTOMERNAME']
                .count()
                .unstack(fill_value=0)
            )
            rolling_year_transactions['Total_Transactions'] = rolling_year_transactions.sum(axis=1)
            seasonality_indices = rolling_year_transactions.div(rolling_year_transactions['Total_Transactions'], axis=0)
            seasonality_indices.drop(columns=['Total_Transactions'], inplace=True)
            seasonality_indices.columns = [f'Seasonality_Index_{col}' for col in seasonality_indices.columns]
            df = df.merge(seasonality_indices, how='left', left_on='CUSTOMERNAME', right_index=True)
            seasonality_columns = [col for col in df.columns if col.startswith('Seasonality_Index_')]
            df[seasonality_columns] = df[seasonality_columns].fillna(0)
            return df

        engineered_data = calculate_seasonality_indices(engineered_data)

        # Aggregation Function
        def aggregate_features_by_customer(df):
            if 'CUSTOMERNAME' not in df.columns:
                st.error("Error: 'CUSTOMERNAME' column is required for aggregation.")
                return df

            agg_df = df.groupby('CUSTOMERNAME').agg(
                Total_Amount_Purchased=('AMOUNT', lambda x: x[x > 0].sum()),
                Maximum_Amount_Purchased=('AMOUNT', lambda x: x[x > 0].max()),
                Minimum_Amount_Purchased=('AMOUNT', lambda x: x[x > 0].min()),
                Total_Volume_Purchased=('VOLUME', lambda x: x[x > 0].sum()),
                Maximum_Volume_Purchased=('VOLUME', lambda x: x[x > 0].max()),
                Minimum_Volume_Purchased=('VOLUME', lambda x: x[x > 0].min()),
                Most_Frequent_Item_Group=('ITEMGROUPDESCRIPTION', lambda x: x.mode().iloc[0] if not x.mode().empty else None),
                Customer_Transactions=('Customer_Transactions', 'first'),
                Months_Since_Last_Purchase=('Months_Since_Last_Purchase', 'first'),
                Customer_Lifetime=('Customer_Lifetime', 'first'),
                Active_Month_Percentage=('Customer_Lifetime', lambda x: x[x > 0].count() / x.count()),
                Std_Dev_Time_Between_Purchases=('Std_Dev_Time_Between_Purchases', 'first'),
                Mean_Time_Between_Purchases=('Mean_Time_Between_Purchases', 'first'),
                Max_Time_Without_Purchase=('Max_Time_Without_Purchase', 'first'),
                Is_Repeat_Customer=('Is_Repeat_Customer', 'first'),
                Average_Purchase_Value=('AMOUNT', lambda x: x[x > 0].sum() / len(x)),
                Total_Refund_Amount=('AMOUNT', lambda x: x[x < 0].sum()),
                First_Purchase=('First_Purchase', 'first'),
                Last_Purchase=('Last_Purchase', 'first'),
                **{col: (col, 'first') for col in df.columns if col.startswith('Seasonality_Index_')}
            ).reset_index()

            # Derived metrics
            agg_df['Refund_Ratio'] = agg_df['Total_Refund_Amount'].abs() / (
                agg_df['Total_Amount_Purchased'] + agg_df['Total_Refund_Amount'].abs())
            agg_df['Average_Purchase_Per_Month'] = agg_df.apply(
                lambda x: x['Total_Amount_Purchased'] / x['Customer_Lifetime'] if x['Customer_Lifetime'] > 0 else 0, axis=1
            )
            agg_df['Purchase_Frequency_Per_Month'] = agg_df.apply(
                lambda x: x['Customer_Transactions'] / x['Customer_Lifetime'] if x['Customer_Lifetime'] > 0 else 0, axis=1
            )
            return agg_df

        aggregated_data = aggregate_features_by_customer(engineered_data)

        st.write("### Aggregated Customer Data:")
        st.dataframe(aggregated_data.head())

        # Data Download as Excel
        def convert_to_excel(df):
            output = BytesIO()
            with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                df.to_excel(writer, index=False, sheet_name='Aggregated Data')
            return output.getvalue()

        excel_data = convert_to_excel(aggregated_data)

        # Streamlit download button
        st.download_button(
            label="Download Aggregated Data as Excel",
            data=excel_data,
            file_name="aggregated_data.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

    except Exception as e:
        # Handle errors gracefully
        st.error(f"An error occurred: {e}")
