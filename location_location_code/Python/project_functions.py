import pandas as pd
import numpy as np
from sklearn.preprocessing import scale


def sales_df_reformatting(sales_df):
    sales_df = sales_df.fillna(0)
    sales_df = sales_df.drop_duplicates(subset='store_code', keep='last')
    DATE = pd.to_datetime(sales_df.columns.values.tolist()[1:])
    sales_mx = sales_df.iloc[:, 1:].transpose()
    sales_mx.columns = sales_df['store_code']
    sales_mx['DATE'] = DATE
    sales_mx['DATE'] = sales_mx['DATE'].dt.year.apply(str) + sales_mx['DATE'].dt.week.apply(str)
    sales_mx = sales_mx.groupby('DATE')
    new_sales_df = sales_mx.aggregate('sum')
    return new_sales_df


def create_target_variables_df(sales_df):
    avg_sales = sales_df.tail(12).apply('mean')
    sales_sd = sales_df.tail(12).apply('std')

    sales_data_df = pd.DataFrame(
        {'STORE_CODE': sales_df.columns.values,
         'AVG_SALES': avg_sales,
         'SALES_VOLATILITY': sales_sd}
    )
    outliers_filter = (abs(scale(sales_data_df['AVG_SALES'])) < 2)
    sales_data_df = sales_data_df[outliers_filter & (sales_data_df['AVG_SALES'] > 0)]
    return sales_data_df


def extract_counts(dictionary):
    return [len(dictionary[x]) for x in dictionary.keys()]


def create_amenities_count_df(store_codes_json,
                              surroundings_json,
                              amenities_ls):
    store_code_amenities_counts = [extract_counts(surroundings_json[n]['surroundings']) for n in
                                   range(len(surroundings_json))]
    store_code_amenities_counts = np.asarray(store_code_amenities_counts)
    store_code_amenities_df = pd.DataFrame(store_code_amenities_counts,
                                           index=store_codes_json,
                                           columns=amenities_ls)
    return store_code_amenities_df
