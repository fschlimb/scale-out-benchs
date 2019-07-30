# Derived from https://github.com/rapidsai/notebooks-extended/tree/master/intermediate_notebooks/E2E/mortgage

import numpy as np
import pygdf
from pygdf.dataframe import DataFrame
from collections import OrderedDict
from glob import glob
import os
import time


def null_workaround(df, **kwargs):
    for column, data_type in df.dtypes.items():
        if str(data_type) == "category":
            df[column] = df[column].astype('int32').fillna(-1)
        if str(data_type) in ['int8', 'int16', 'int32', 'int64', 'float32', 'float64']:
            df[column] = df[column].fillna(-1)
    return df

def run_gpu_workflow(quarter=1, year=2000, perf_file="", **kwargs):
    t1 = time.time()
    names = gpu_load_names()
    acq_gdf = gpu_load_acquisition_csv(acquisition_path="/rapids/data/mortgage/acq/Acquisition_"
                                      + str(year) + "Q" + str(quarter) + ".txt")
    perf_df_tmp = gpu_load_performance_csv(perf_file)
    print("read time", time.time()-t1)
    t1 = time.time()
    acq_gdf = acq_gdf.merge(names, how='left', on=['seller_name'])
    acq_gdf.drop_column('seller_name')
    acq_gdf['seller_name'] = acq_gdf['new']
    acq_gdf.drop_column('new')
    gdf = perf_df_tmp
    everdf = create_ever_features(gdf)
    delinq_merge = create_delinq_features(gdf)
    everdf = join_ever_delinq_features(everdf, delinq_merge)
    del(delinq_merge)
    joined_df = create_joined_df(gdf, everdf)
    testdf = create_12_mon_features(joined_df)
    joined_df = combine_joined_12_mon(joined_df, testdf)
    del(testdf)
    perf_df = final_performance_delinquency(gdf, joined_df)
    del(gdf, joined_df)
    final_gdf = join_perf_acq_gdfs(perf_df, acq_gdf)
    del(perf_df)
    del(acq_gdf)
    print("compute time", time.time()-t1)
    final_gdf = last_mile_cleaning(final_gdf)
    print("compute time with copy to host", time.time()-t1)
    return final_gdf

def gpu_load_performance_csv(performance_path, **kwargs):
    """ Loads performance data

    Returns
    -------
    GPU DataFrame
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

    dtypes = OrderedDict([
        ("loan_id", "int64"),
        ("monthly_reporting_period", "date"),
        ("servicer", "category"),
        ("interest_rate", "float64"),
        ("current_actual_upb", "float64"),
        ("loan_age", "float64"),
        ("remaining_months_to_legal_maturity", "float64"),
        ("adj_remaining_months_to_maturity", "float64"),
        ("maturity_date", "date"),
        ("msa", "float64"),
        ("current_loan_delinquency_status", "int32"),
        ("mod_flag", "category"),
        ("zero_balance_code", "category"),
        ("zero_balance_effective_date", "date"),
        ("last_paid_installment_date", "date"),
        ("foreclosed_after", "date"),
        ("disposition_date", "date"),
        ("foreclosure_costs", "float64"),
        ("prop_preservation_and_repair_costs", "float64"),
        ("asset_recovery_costs", "float64"),
        ("misc_holding_expenses", "float64"),
        ("holding_taxes", "float64"),
        ("net_sale_proceeds", "float64"),
        ("credit_enhancement_proceeds", "float64"),
        ("repurchase_make_whole_proceeds", "float64"),
        ("other_foreclosure_proceeds", "float64"),
        ("non_interest_bearing_upb", "float64"),
        ("principal_forgiveness_upb", "float64"),
        ("repurchase_make_whole_proceeds_flag", "category"),
        ("foreclosure_principal_write_off_amount", "float64"),
        ("servicing_activity_indicator", "category")
    ])

    print(performance_path)

    return pygdf.read_csv(performance_path, names=cols, delimiter='|', dtype=list(dtypes.values()), skiprows=1)

def gpu_load_acquisition_csv(acquisition_path, **kwargs):
    """ Loads acquisition data

    Returns
    -------
    GPU DataFrame
    """

    cols = [
        'loan_id', 'orig_channel', 'seller_name', 'orig_interest_rate', 'orig_upb', 'orig_loan_term',
        'orig_date', 'first_pay_date', 'orig_ltv', 'orig_cltv', 'num_borrowers', 'dti', 'borrower_credit_score',
        'first_home_buyer', 'loan_purpose', 'property_type', 'num_units', 'occupancy_status', 'property_state',
        'zip', 'mortgage_insurance_percent', 'product_type', 'coborrow_credit_score', 'mortgage_insurance_type',
        'relocation_mortgage_indicator'
    ]

    dtypes = OrderedDict([
        ("loan_id", "int64"),
        ("orig_channel", "category"),
        ("seller_name", "category"),
        ("orig_interest_rate", "float64"),
        ("orig_upb", "int64"),
        ("orig_loan_term", "int64"),
        ("orig_date", "date"),
        ("first_pay_date", "date"),
        ("orig_ltv", "float64"),
        ("orig_cltv", "float64"),
        ("num_borrowers", "float64"),
        ("dti", "float64"),
        ("borrower_credit_score", "float64"),
        ("first_home_buyer", "category"),
        ("loan_purpose", "category"),
        ("property_type", "category"),
        ("num_units", "int64"),
        ("occupancy_status", "category"),
        ("property_state", "category"),
        ("zip", "int64"),
        ("mortgage_insurance_percent", "float64"),
        ("product_type", "category"),
        ("coborrow_credit_score", "float64"),
        ("mortgage_insurance_type", "float64"),
        ("relocation_mortgage_indicator", "category")
    ])

    print(acquisition_path)

    return pygdf.read_csv(acquisition_path, names=cols, delimiter='|', dtype=list(dtypes.values()), skiprows=1)

def gpu_load_names(**kwargs):
    """ Loads names used for renaming the banks

    Returns
    -------
    GPU DataFrame
    """

    cols = [
        'seller_name', 'new'
    ]

    dtypes = OrderedDict([
        ("seller_name", "category"),
        ("new", "category"),
    ])

    return pygdf.read_csv("/rapids/data/mortgage/names.csv", names=cols, delimiter='|', dtype=list(dtypes.values()), skiprows=1)




def create_ever_features(gdf, **kwargs):
    everdf = gdf[['loan_id', 'current_loan_delinquency_status']]
    everdf = everdf.groupby('loan_id', method='hash').max()
    del(gdf)
    everdf['ever_30'] = (everdf['max_current_loan_delinquency_status'] >= 1).astype('int8')
    everdf['ever_90'] = (everdf['max_current_loan_delinquency_status'] >= 3).astype('int8')
    everdf['ever_180'] = (everdf['max_current_loan_delinquency_status'] >= 6).astype('int8')
    everdf.drop_column('max_current_loan_delinquency_status')
    return everdf


def create_delinq_features(gdf, **kwargs):
    delinq_gdf = gdf[['loan_id', 'monthly_reporting_period', 'current_loan_delinquency_status']]
    del(gdf)
    delinq_30 = delinq_gdf.query('current_loan_delinquency_status >= 1')[['loan_id', 'monthly_reporting_period']].groupby('loan_id', method='hash').min()
    delinq_30['delinquency_30'] = delinq_30['min_monthly_reporting_period']
    delinq_30.drop_column('min_monthly_reporting_period')
    delinq_90 = delinq_gdf.query('current_loan_delinquency_status >= 3')[['loan_id', 'monthly_reporting_period']].groupby('loan_id', method='hash').min()
    delinq_90['delinquency_90'] = delinq_90['min_monthly_reporting_period']
    delinq_90.drop_column('min_monthly_reporting_period')
    delinq_180 = delinq_gdf.query('current_loan_delinquency_status >= 6')[['loan_id', 'monthly_reporting_period']].groupby('loan_id', method='hash').min()
    delinq_180['delinquency_180'] = delinq_180['min_monthly_reporting_period']
    delinq_180.drop_column('min_monthly_reporting_period')
    del(delinq_gdf)
    delinq_merge = delinq_30.merge(delinq_90, how='left', on=['loan_id'], type='hash')
    delinq_merge['delinquency_90'] = delinq_merge['delinquency_90'].fillna(np.dtype('datetime64[ms]').type('1970-01-01').astype('datetime64[ms]'))
    delinq_merge = delinq_merge.merge(delinq_180, how='left', on=['loan_id'], type='hash')
    delinq_merge['delinquency_180'] = delinq_merge['delinquency_180'].fillna(np.dtype('datetime64[ms]').type('1970-01-01').astype('datetime64[ms]'))
    del(delinq_30)
    del(delinq_90)
    del(delinq_180)
    return delinq_merge



def join_ever_delinq_features(everdf_tmp, delinq_merge, **kwargs):
    everdf = everdf_tmp.merge(delinq_merge, on=['loan_id'], how='left', type='hash')
    del(everdf_tmp)
    del(delinq_merge)
    everdf['delinquency_30'] = everdf['delinquency_30'].fillna(np.dtype('datetime64[ms]').type('1970-01-01').astype('datetime64[ms]'))
    everdf['delinquency_90'] = everdf['delinquency_90'].fillna(np.dtype('datetime64[ms]').type('1970-01-01').astype('datetime64[ms]'))
    everdf['delinquency_180'] = everdf['delinquency_180'].fillna(np.dtype('datetime64[ms]').type('1970-01-01').astype('datetime64[ms]'))
    return everdf




def create_joined_df(gdf, everdf, **kwargs):
    test = gdf[['loan_id', 'monthly_reporting_period', 'current_loan_delinquency_status', 'current_actual_upb']]
    del(gdf)
    test['timestamp'] = test['monthly_reporting_period']
    test.drop_column('monthly_reporting_period')
    test['timestamp_month'] = test['timestamp'].dt.month
    test['timestamp_year'] = test['timestamp'].dt.year
    test['delinquency_12'] = test['current_loan_delinquency_status']
    test.drop_column('current_loan_delinquency_status')
    test['upb_12'] = test['current_actual_upb']
    test.drop_column('current_actual_upb')
    test['upb_12'] = test['upb_12'].fillna(999999999)
    test['delinquency_12'] = test['delinquency_12'].fillna(-1)

    joined_df = test.merge(everdf, how='left', on=['loan_id'], type='hash')
    del(everdf)
    del(test)

    joined_df['ever_30'] = joined_df['ever_30'].fillna(-1)
    joined_df['ever_90'] = joined_df['ever_90'].fillna(-1)
    joined_df['ever_180'] = joined_df['ever_180'].fillna(-1)
    joined_df['delinquency_30'] = joined_df['delinquency_30'].fillna(-1)
    joined_df['delinquency_90'] = joined_df['delinquency_90'].fillna(-1)
    joined_df['delinquency_180'] = joined_df['delinquency_180'].fillna(-1)

    joined_df['timestamp_year'] = joined_df['timestamp_year'].astype('int32')
    joined_df['timestamp_month'] = joined_df['timestamp_month'].astype('int32')

    return joined_df



def create_12_mon_features(joined_df, **kwargs):
    testdfs = []
    n_months = 12
    for y in range(1, n_months + 1):
        tmpdf = joined_df[['loan_id', 'timestamp_year', 'timestamp_month', 'delinquency_12', 'upb_12']]
        tmpdf['josh_months'] = tmpdf['timestamp_year'] * 12 + tmpdf['timestamp_month']
        tmpdf['josh_mody_n'] = ((tmpdf['josh_months'].astype('float64') - 24000 - y) / 12).floor()
        tmpdf = tmpdf.groupby(['loan_id', 'josh_mody_n'], method='hash').agg({'delinquency_12': 'max','upb_12': 'min'})
        tmpdf['delinquency_12'] = (tmpdf['max_delinquency_12']>3).astype('int32')
        tmpdf['delinquency_12'] +=(tmpdf['min_upb_12']==0).astype('int32')
        tmpdf.drop_column('max_delinquency_12')
        tmpdf['upb_12'] = tmpdf['min_upb_12']
        tmpdf.drop_column('min_upb_12')
        tmpdf['timestamp_year'] = (((tmpdf['josh_mody_n'] * n_months) + 24000 + (y - 1)) / 12).floor().astype('int16')
        tmpdf['timestamp_month'] = np.int8(y)
        tmpdf.drop_column('josh_mody_n')
        testdfs.append(tmpdf)
        del(tmpdf)
    del(joined_df)

    return pygdf.concat(testdfs)


def combine_joined_12_mon(joined_df, testdf, **kwargs):
    joined_df.drop_column('delinquency_12')
    joined_df.drop_column('upb_12')
    joined_df['timestamp_year'] = joined_df['timestamp_year'].astype('int16')
    joined_df['timestamp_month'] = joined_df['timestamp_month'].astype('int8')
    return joined_df.merge(testdf, how='left', on=['loan_id', 'timestamp_year', 'timestamp_month'], type='hash')


def final_performance_delinquency(gdf, joined_df, **kwargs):
    merged = null_workaround(gdf)
    joined_df = null_workaround(joined_df)
    merged['timestamp_month'] = merged['monthly_reporting_period'].dt.month
    merged['timestamp_month'] = merged['timestamp_month'].astype('int8')
    merged['timestamp_year'] = merged['monthly_reporting_period'].dt.year
    merged['timestamp_year'] = merged['timestamp_year'].astype('int16')
    merged = merged.merge(joined_df, how='left', on=['loan_id', 'timestamp_year', 'timestamp_month'], type='hash')
    merged.drop_column('timestamp_year')
    merged.drop_column('timestamp_month')
    return merged



def join_perf_acq_gdfs(perf, acq, **kwargs):
    perf = null_workaround(perf)
    acq = null_workaround(acq)
    return perf.merge(acq, how='left', on=['loan_id'], type='hash')



def last_mile_cleaning(df, **kwargs):
    for col, dtype in df.dtypes.iteritems():
        if str(dtype)=='category':
            df[col] = df[col].cat.codes
    df['delinquency_12'] = df['delinquency_12'] > 0
    df['delinquency_12'] = df['delinquency_12'].fillna(False).astype('int32')
    return df.to_arrow(index=False)


data_directory = "/rapids/data/mortgage/perf/"
year = 2000
quarter = 1
perf_file = os.path.join(data_directory, "Performance_" + str(year) + "Q" + str(quarter) + ".txt")
#gdf = run_gpu_workflow(year=year, quarter=quarter, perf_file=perf_file)
#print(gdf)
t1 = time.time()
gdf = run_gpu_workflow(year=year, quarter=quarter, perf_file=perf_file)
t2 = time.time()
print("Total exec time:", t2-t1)


# start_year = 2000
# end_year = 2017

# gpu_dfs = []
# gpu_time = 0
# quarter = 1
# year = start_year
# while year != end_year:
#     for file in glob(os.path.join(data_directory, "Performance_" + str(year) + "Q" + str(quarter) + "*")):
#         gpu_dfs.append(process_quarter_gpu(year=year, quarter=quarter, perf_file=file))
#     quarter += 1
#     if quarter == 5:
#         year += 1
#         quarter = 1
# wait(gpu_dfs)
