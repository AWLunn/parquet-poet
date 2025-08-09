import pandas as pd

def load_data(filepath):
    return pd.read_excel(filepath)

def clean_data(df):
    df.columns = df.columns.str.strip().str.lower()

    # Fix domain typos and normalize case
    df['domain'] = df['domain'].str.upper().str.strip()
    df['domain'] = df['domain'].replace({'RESTRAUNT': 'RESTAURANT'})

    # Standardize location formatting
    df['location'] = df['location'].str.strip().str.title()

    # Type conversions
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df['value'] = df['value'].astype(float)
    df['transaction_count'] = df['transaction_count'].astype(int)

    return df

def aggregate_data(df):
    # Domain averages by date
    df_domain = df.groupby(['date', 'domain']).agg({
        'value': 'mean',
        'transaction_count': 'mean'
    }).reset_index()

    # Counts and averages by city
    df_city = df.groupby('location').agg({
        'value': ['mean', 'sum']
    }).reset_index()
    df_city.columns = ['location', 'value_mean', 'value_sum']

    # Daily totals
    df_daily = df.groupby('date').agg({
        'value': 'sum',
        'transaction_count': 'sum'
    }).reset_index()

    return df_domain, df_city, df_daily


def save_outputs(df_domain, df_city, df_daily):
    df_domain.to_parquet("../data/domain_aggregates.parquet")
    df_city.to_parquet("../data/city_aggregates.parquet")
    df_daily.to_parquet("../data/daily_totals.parquet")

if __name__ == "__main__":
    df = load_data("../data/raw/bankdataset.xlsx")
    df_cleaned = clean_data(df)
    df_domain, df_city, df_daily = aggregate_data(df_cleaned)
    print("💡 domain columns before save:", df_domain.columns.tolist())

    save_outputs(df_domain, df_city, df_daily)
