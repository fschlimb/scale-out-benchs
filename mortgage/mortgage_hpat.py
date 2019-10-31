# Derived from https://github.com/rapidsai/notebooks-extended/tree/master/intermediate_notebooks/E2E/mortgage

# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# You need a relatively recent HPAT/SDC which supports boxing cotegoricals
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

import numpy as np
import pandas as pd
from pandas.api.types import CategoricalDtype
from collections import OrderedDict
from glob import glob
import os
import time
import hpat


def run_pd_workflow(quarter=1, year=2000, perf_file=""):
    t1 = time.time()
    names = pd_load_names()
    acq_pdf = pd_load_acquisition_csv(acquisition_path="mortgage/acq/Acquisition_"
                                      + str(year) + "Q" + str(quarter) + ".txt")
    perf_df_tmp = pd_load_performance_csv(perf_file)
    print("read time", time.time()-t1)
    t1 = time.time()
    acq_pdf = acq_pdf.merge(names, how='left', on=['seller_name'])
    acq_pdf.drop(columns=['seller_name'], inplace=True)
    acq_pdf['seller_name'] = acq_pdf['new']
    acq_pdf.drop(columns=['new'], inplace=True)
    pdf = perf_df_tmp
    everdf = create_ever_features(pdf)
    delinq_merge = create_delinq_features(pdf)
    everdf = join_ever_delinq_features(everdf, delinq_merge)
    del(delinq_merge)
    joined_df = create_joined_df(pdf, everdf)
    testdf = create_12_mon_features(joined_df)
    joined_df = combine_joined_12_mon(joined_df, testdf)
    del(testdf)
    perf_df = final_performance_delinquency(pdf, joined_df)
    del(pdf, joined_df)
    final_pdf = join_perf_acq_pdfs(perf_df, acq_pdf)
    del(perf_df)
    del(acq_pdf)
    print("compute time", time.time()-t1)
    final_pdf = last_mile_cleaning(final_pdf)
    print("compute time with copy to host", time.time()-t1)
    return final_pdf


@hpat.jit(distributed=['df'])
def pd_load_performance_csv(performance_path):
    """ Loads performance data

    Returns
    -------
    PD DataFrame
    """

    cols = [
        "loan_id", "monthly_reporting_period", "servicer", "interest_rate", "current_actual_upb",
        "loan_age", "remaining_months_to_legal_maturity", "adj_remaining_months_to_maturity",
        "maturity_date", "msa", "current_loan_delinquency_status", "mod_flag", "zero_balance_code",
        "zero_balance_effective_date", "last_paid_installment_date", "foreclosed_after",
        "disposition_date", "foreclosure_costs", "prop_preservation_and_repair_costs",
        "asset_recovery_costs", "misc_holding_expenses", "holding_taxes", "net_sale_proceeds",
        "credit_enhancement_proceeds", "repurchase_make_whole_proceeds", "other_foreclosure_proceeds",
        "non_interest_bearing_upb", "principal_forgiveness_upb", "repurchase_make_whole_proceeds_flag",
        "foreclosure_principal_write_off_amount", "servicing_activity_indicator"
    ]
    dtypes = {
        "loan_id": np.int64,
        "monthly_reporting_period": str,
        "servicer": str,
        "interest_rate": np.float64,
        "current_actual_upb": np.float64,
        "loan_age": np.float64,
        "remaining_months_to_legal_maturity": np.float64,
        "adj_remaining_months_to_maturity": np.float64,
        "maturity_date": str,
        "msa": np.float64,
        "current_loan_delinquency_status": np.int32,
        "mod_flag": CategoricalDtype(['N', 'Y']),
        "zero_balance_code": CategoricalDtype(['01', '02', '06', '09', '03', '15', '16']),
        "zero_balance_effective_date": str,
        "last_paid_installment_date": str,
        "foreclosed_after": str,
        "disposition_date": str,
        "foreclosure_costs": np.float64,
        "prop_preservation_and_repair_costs": np.float64,
        "asset_recovery_costs": np.float64,
        "misc_holding_expenses": np.float64,
        "holding_taxes": np.float64,
        "net_sale_proceeds": np.float64,
        "credit_enhancement_proceeds": np.float64,
        "repurchase_make_whole_proceeds": np.float64,
        "other_foreclosure_proceeds": np.float64,
        "non_interest_bearing_upb": np.float64,
        "principal_forgiveness_upb": np.float64,
        "repurchase_make_whole_proceeds_flag": CategoricalDtype(['N', 'Y']),
        "foreclosure_principal_write_off_amount": np.float64,
        "servicing_activity_indicator": CategoricalDtype(['N', 'Y']),
    }

    print(performance_path)

    #return pd.read_csv(performance_path, names=cols, delimiter='|', dtype=dtypes, parse_dates=[1,8,13,14,15,16])
    df = pd.read_csv('perf.csv', names=cols, delimiter='|', dtype=dtypes, parse_dates=[1,8,13,14,15,16])
    return df


@hpat.jit(distributed=['df'])
def pd_load_acquisition_csv(acquisition_path):
    """ Loads acquisition data

    Returns
    -------
    PD DataFrame
    """

    cols = [
        'loan_id', 'orig_channel', 'seller_name', 'orig_interest_rate', 'orig_upb', 'orig_loan_term',
        'orig_date', 'first_pay_date', 'orig_ltv', 'orig_cltv', 'num_borrowers', 'dti', 'borrower_credit_score',
        'first_home_buyer', 'loan_purpose', 'property_type', 'num_units', 'occupancy_status', 'property_state',
        'zip', 'mortgage_insurance_percent', 'product_type', 'coborrow_credit_score', 'mortgage_insurance_type',
        'relocation_mortgage_indicator'
    ]
    dtypes = {
        "loan_id": np.int64,
        "orig_channel": CategoricalDtype(['B', 'C', 'R']),
        "seller_name": str,
        "orig_interest_rate": np.float64,
        "orig_upb": np.int64,
        "orig_loan_term": np.int64,
        "orig_date": str,
        "first_pay_date": str,
        "orig_ltv": np.float64,
        "orig_cltv": np.float64,
        "num_borrowers": np.float64,
        "dti": np.float64,
        "borrower_credit_score": np.float64,
        "first_home_buyer": CategoricalDtype(['N', 'U', 'Y']),
        "loan_purpose": CategoricalDtype(['C', 'P', 'R', 'U']),
        "property_type": CategoricalDtype(['CO', 'CP', 'MH', 'PU', 'SF']),
        "num_units": np.int64,
        "occupancy_status": CategoricalDtype(['I', 'P', 'S']),
        "property_state": CategoricalDtype(
            ['AK', 'AL', 'AR', 'AZ', 'CA', 'CO', 'CT', 'DC', 'DE', 'FL', 'GA', 'HI',
            'IA', 'ID', 'IL', 'IN', 'KS', 'KY', 'LA', 'MA', 'MD', 'ME', 'MI', 'MN',
            'MO', 'MS', 'MT', 'NC', 'ND', 'NE', 'NH', 'NJ', 'NM', 'NV', 'NY', 'OH',
            'OK', 'OR', 'PA', 'PR', 'RI', 'SC', 'SD', 'TN', 'TX', 'UT', 'VA', 'VI',
            'VT', 'WA', 'WI', 'WV', 'WY']),
        "zip": np.int64,
        "mortgage_insurance_percent": np.float64,
        "product_type": CategoricalDtype(['FRM']),
        "coborrow_credit_score": np.float64,
        "mortgage_insurance_type": np.float64,
        "relocation_mortgage_indicator": CategoricalDtype(['N', 'Y']),
    }
    print(acquisition_path)

    #return pd.read_csv(acquisition_path, names=cols, delimiter='|', dtype=dtypes, parse_dates=[6,7])
    df = pd.read_csv('acq.csv', names=cols, delimiter='|', dtype=dtypes, parse_dates=[6,7])
    return df


@hpat.jit(distributed=['df'])
def pd_load_names():
    """ Loads names used for renaming the banks

    Returns
    -------
    PD DataFrame
    """

    cols = ['seller_name', 'new']
    dtypes = {'seller_name':str, 'new':str}

    df = pd.read_csv("mortgage/names.csv", names=cols, delimiter='|', dtype=dtypes)
    return df


@hpat.jit
def create_ever_features(pdf):
    everdf = pdf[['loan_id', 'current_loan_delinquency_status']]
    everdf = everdf.groupby('loan_id', as_index=False).max()
    #del(pdf)
    everdf['ever_30'] = (everdf['current_loan_delinquency_status'] >= 1).astype(np.int8)
    everdf['ever_90'] = (everdf['current_loan_delinquency_status'] >= 3).astype(np.int8)
    everdf['ever_180'] = (everdf['current_loan_delinquency_status'] >= 6).astype(np.int8)
    everdf.drop(columns=['current_loan_delinquency_status'], inplace=True)
    return everdf


# for some reason this cannot be in compiled code
# HPAT currently doesn't seem to support datetime64[ms], using [ns]
dt64_fill = np.dtype('datetime64[ns]').type('1970-01-01').astype('datetime64[ns]')

@hpat.jit
def create_delinq_features(pdf):
    delinq_df = pdf[['loan_id', 'monthly_reporting_period', 'current_loan_delinquency_status']]
    #del(pdf)
    delinq_30 = delinq_df[delinq_df['current_loan_delinquency_status'] >= 1][['loan_id', 'monthly_reporting_period']].groupby('loan_id', as_index=False).min()
    delinq_30['delinquency_30'] = delinq_30['monthly_reporting_period']
    delinq_30.drop(columns=['monthly_reporting_period'], inplace=True)
    delinq_90 = delinq_df[delinq_df['current_loan_delinquency_status'] >= 3][['loan_id', 'monthly_reporting_period']].groupby('loan_id', as_index=False).min()
    delinq_90['delinquency_90'] = delinq_90['monthly_reporting_period']
    delinq_90.drop(columns=['monthly_reporting_period'], inplace=True)
    delinq_180 = delinq_df[delinq_df['current_loan_delinquency_status'] >= 6][['loan_id', 'monthly_reporting_period']].groupby('loan_id', as_index=False).min()
    delinq_180['delinquency_180'] = delinq_180['monthly_reporting_period']
    delinq_180.drop(columns=['monthly_reporting_period'], inplace=True)
    #del(delinq_df)
    delinq_merge = delinq_30.merge(delinq_90, how='left', on=['loan_id'])
    delinq_merge['delinquency_90'] = delinq_merge['delinquency_90'].fillna(dt64_fill)
    delinq_merge = delinq_merge.merge(delinq_180, how='left', on=['loan_id'])
    delinq_merge['delinquency_180'] = delinq_merge['delinquency_180'].fillna(dt64_fill)
    #del(delinq_30)
    #del(delinq_90)
    #del(delinq_180)
    return delinq_merge


@hpat.jit
def join_ever_delinq_features(everdf_tmp, delinq_merge):
    everdf = everdf_tmp.merge(delinq_merge, on=['loan_id'], how='left')
    #del(everdf_tmp)
    #del(delinq_merge)
    everdf['delinquency_30'] = everdf['delinquency_30'].fillna(dt64_fill)
    everdf['delinquency_90'] = everdf['delinquency_90'].fillna(dt64_fill)
    everdf['delinquency_180'] = everdf['delinquency_180'].fillna(dt64_fill)
    return everdf


@hpat.jit(distributed=['joined_df'])
def create_joined_df(pdf, everdf):
    test = pdf[['loan_id', 'monthly_reporting_period', 'current_loan_delinquency_status', 'current_actual_upb']]
    #del(pdf)
    test['timestamp'] = test['monthly_reporting_period']
    test.drop(columns=['monthly_reporting_period'], inplace=True)
    test['timestamp_month'] = test['timestamp'].dt.month
    test['timestamp_year'] = test['timestamp'].dt.year
    test['delinquency_12'] = test['current_loan_delinquency_status']
    test.drop(columns=['current_loan_delinquency_status'], inplace=True)
    test['upb_12'] = test['current_actual_upb']
    test.drop(columns=['current_actual_upb'], inplace=True)
    test['upb_12'] = test['upb_12'].fillna(999999999)
    test['delinquency_12'] = test['delinquency_12'].fillna(-1)

    joined_df = test.merge(everdf, how='left', on=['loan_id'])
    #del(everdf)
    #del(test)

    joined_df['ever_30'] = joined_df['ever_30'].fillna(-1)
    joined_df['ever_90'] = joined_df['ever_90'].fillna(-1)
    joined_df['ever_180'] = joined_df['ever_180'].fillna(-1)
    joined_df['delinquency_30'] = joined_df['delinquency_30'].fillna(dt64_fill)
    joined_df['delinquency_90'] = joined_df['delinquency_90'].fillna(dt64_fill)
    joined_df['delinquency_180'] = joined_df['delinquency_180'].fillna(dt64_fill)

    joined_df['timestamp_year'] = joined_df['timestamp_year'].astype(np.int32)
    joined_df['timestamp_month'] = joined_df['timestamp_month'].astype(np.int32)

    return joined_df


@hpat.jit(distributed=['df'])
def create_12_mon_features(joined_df):
    #testdfs = []
    n_months = 12
    for y in range(1, n_months + 1):
        tmpdf = joined_df[['loan_id', 'timestamp_year', 'timestamp_month', 'delinquency_12', 'upb_12']]
        tmpdf['josh_months'] = tmpdf['timestamp_year'] * 12 + tmpdf['timestamp_month']
        tmpdf['josh_mody_n'] = np.floor((tmpdf['josh_months'].astype(np.float64) - 24000 - y) / 12)
        # tmpdf = tmpdf.groupby(['loan_id', 'josh_mody_n'], as_index=False).agg({'delinquency_12': 'max','upb_12': 'min'})
        tmpdf_d = tmpdf.groupby(['loan_id', 'josh_mody_n'], as_index=False)['delinquency_12'].max()
        tmpdf_m = tmpdf.groupby(['loan_id', 'josh_mody_n'], as_index=False)['upb_12'].min()
        tmpdf_d['upb_12'] = tmpdf_m['upb_12']
        tmpdf2 = tmpdf_d
        tmpdf2['delinquency_12'] = (tmpdf2['delinquency_12']>3).astype(np.int32)
        tmpdf2['delinquency_12'] +=(tmpdf2['upb_12']==0).astype(np.int32)
        tmpdf2['timestamp_year'] = np.floor(((tmpdf2['josh_mody_n'] * n_months) + 24000 + (y - 1)) / 12).astype(np.int16)
        tmpdf2['timestamp_month'] = np.full_like(tmpdf2['timestamp_year'].values, y, np.int8)
        #tmpdf2['timestamp_month'] = np.int8(y)
        tmpdf2.drop(columns=['josh_mody_n'], inplace=True)
        #testdfs.append(tmpdf2)
        if y == 1:
            df = tmpdf2
        else:
            tmpdf3 = df.append(tmpdf2, ignore_index=True)
            df['loan_id'] = tmpdf3['loan_id']
            df['timestamp_year'] = tmpdf3['timestamp_year']
            df['timestamp_month'] = tmpdf3['timestamp_month']
            df['delinquency_12'] = tmpdf3['delinquency_12']
            df['upb_12'] = tmpdf3['upb_12']
            
        #del(tmpdf2)
    #del(joined_df)

    return df #pd.concat(testdfs) # aaaahhhh


@hpat.jit(distributed=['joined_df'])
def combine_joined_12_mon(joined_df, testdf):
    joined_df.drop(columns=['delinquency_12', 'upb_12'], inplace=True)
    joined_df['timestamp_year'] = joined_df['timestamp_year'].astype(np.int16)
    joined_df['timestamp_month'] = joined_df['timestamp_month'].astype(np.int8)
    return joined_df.merge(testdf, how='left', on=['loan_id', 'timestamp_year', 'timestamp_month'])


@hpat.jit(distributed=['merged'])
def final_performance_delinquency(merged, joined_df):
    merged['timestamp_month'] = merged['monthly_reporting_period'].dt.month
    merged['timestamp_month'] = merged['timestamp_month'].astype(np.int8)
    merged['timestamp_year'] = merged['monthly_reporting_period'].dt.year
    merged['timestamp_year'] = merged['timestamp_year'].astype(np.int16)
    merged = merged.merge(joined_df, how='left', on=['loan_id', 'timestamp_year', 'timestamp_month'])
    merged.drop(columns=['timestamp_year'], inplace=True)
    merged.drop(columns=['timestamp_month'], inplace=True)
    return merged


@hpat.jit(distributed=['df'])
def join_perf_acq_pdfs(perf, acq):
    df = perf.merge(acq, how='left', on=['loan_id'])
    return df


@hpat.jit
def last_mile_cleaning(df):
    #for col, dtype in df.dtypes.iteritems():
    #    if str(dtype)=='category':
    #        df[col] = df[col].cat.codes
    df['delinquency_12'] = df['delinquency_12'] > 0
    df['delinquency_12'] = df['delinquency_12'].fillna(False).astype(np.int32)
    return df #.to_arrow(index=False)


data_directory = "mortgage/perf/"
year = 2000
quarter = 1
perf_file = os.path.join(data_directory, "Performance_" + str(year) + "Q" + str(quarter) + ".txt")
for i in range(2):
    t1 = time.time()
    pdf = run_pd_workflow(year=year, quarter=quarter, perf_file=perf_file)
    t2 = time.time()
    print("Total exec time1:", t2-t1)


# start_year = 2000
# end_year = 2017

# pd_dfs = []
# pd_time = 0
# quarter = 1
# year = start_year
# while year != end_year:
#     for file in glob(os.path.join(data_directory, "Performance_" + str(year) + "Q" + str(quarter) + "*")):
#         pd_dfs.append(process_quarter_pd(year=year, quarter=quarter, perf_file=file))
#     quarter += 1
#     if quarter == 5:
#         year += 1
#         quarter = 1
# wait(pd_dfs)
