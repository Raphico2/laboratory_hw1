import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
import sys
from sklearn.preprocessing import MinMaxScaler
import joblib
import os


clinical_values = ['HR', 'O2Sat', 'Temp', 'SBP', 'MAP', 'DBP', 'Resp']

lab_variables = ['BUN', 'Calcium', 'Creatinine',
                'Glucose', 'Magnesium', 'Phosphate', 'Potassium', 'Hct', 'Hgb', 'WBC', 'Platelets']

demographical_values = [ 'Age', 'Gender', 'Unit1','HospAdmTime']

to_delete = ['O2Sat_mean', 'O2Sat_max', 'DBP_max', 'Glucose_std', 'Potassium_mean', 'Platelets_mean', 'Unit1',
             'HR_max', 'HR_min', 'O2Sat_min', 'Temp_max', 'SBP_max', 'SBP_min', 'SBP_mean', 'SBP_std', 'DBP_mean',
             'DBP_std',
             'DBP_min', 'MAP_min', 'MAP_max', 'Resp_max', 'BUN_max', 'BUN_min', 'Creatinine_std', 'Creatinine_max',
             'Creatinine_min',
             'Glucose_max', 'Magnesium_max', 'Magnesium_min', 'Phosphate_max', 'Phosphate_min', 'Hct_mean', 'Hct_std',
             'Hct_min', 'Hct_max',
             'Hgb_min', 'Hgb_max', 'WBC_min', 'WBC_max', 'Platelets_min']

def compute_line_and_label(df):
  sequential_count = 0
  sepsis = 0
  row = 0
  column_label = df['SepsisLabel']
  for index, value in enumerate(column_label):
    if value == 1:
      sequential_count += 1
      if sequential_count == 6:
         sepsis = 1
         row = index - 5
    else:
      sequential_count = 0
    row = index
  return sepsis, row

def fill_nan_values(df):
    imputer = KNNImputer(n_neighbors=5)
    df = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)
    return df

def delete_columns(df, col_to_drop):
  df = df.drop(columns=col_to_drop)
  return df


def compute_patient_data(df):
    sepsis, row = compute_line_and_label(df)
    df = df.loc[0:row]
    line = []
    # for the clinical values we take the mean, max, min and std
    for col in clinical_values:
        description = df[col].describe().to_dict()
        try:
            line.append(float(description['mean']))
        except:
            line.append('NaN')

        if description['count'] == 1:
            line.append(0)
        else:
            line.append(float(description['std']))
        try:
            line.append(float(description['max']))
        except:
            line.append('NaN')

        try:
            line.append(float(description['min']))
        except:
            line.append('NaN')

    for col in lab_variables:
        description = df[col].describe().to_dict()
        try:
            line.append(float(description['mean']))
        except:
            line.append(float(description['max']))

        if description['count'] == 1:
            line.append(0)
        else:
            line.append(float(description['std']))

        try:
            line.append(float(description['max']))
        except:
            line.append('NaN')

        try:
            line.append(float(description['min']))
        except:
            line.append('NaN')

    for col in demographical_values:
        description = df[col].describe().to_dict()
        try:
            val = int(description['mean'])
        except:
            val = 0
        if col == 'Unit1':
            unit1 = val
            try:
                unit2 = int(description['mean'])
            except:
                unit2 = 0
            val = np.max([unit1, unit2])
        line.append(val)

    line.append(int(sepsis))
    return line


dataset = []

def preprocess(file_name):

    columns_name = []
    for col in clinical_values:
        columns_name.append(col + '_mean')
        columns_name.append(col + '_std')
        columns_name.append(col + '_max')
        columns_name.append(col + '_min')

    for col in lab_variables:
        columns_name.append(col + '_mean')
        columns_name.append(col + '_std')
        columns_name.append(col + '_max')
        columns_name.append(col + '_min')

    for col in demographical_values:
        columns_name.append(col)
    columns_name.append('label')

    dataset = []
    file_list_test = [file for file in os.listdir(file_name) if file.endswith('.psv')]
    num_files = len(file_list_test)
    for i in range(0, num_files):
        df = pd.read_csv(file_name+'/patient_' + str(i) + '.psv', sep='|')
        dataset.append(compute_patient_data(df))
    test_df = pd.DataFrame(dataset, columns=columns_name)
    test_df = delete_columns(test_df, to_delete)
    test_df = fill_nan_values(test_df)
    y_test = test_df["label"]
    X_test = test_df.drop("label", axis=1)
    return X_test, y_test, num_files


if __name__ == '__main__':

    file_path = sys.argv[1]
    X_test, y_test, num_patient = preprocess(file_path)
    model = joblib.load('model_gradient_boost.sav')
    y_predicted = model.predict(X_test)
    Y_predicted_int = []
    for i in range(0, len(y_predicted)):
        Y_predicted_int.append(int(y_predicted[0]))


    patient = []
    for i in range(0, num_patient):
        patient.append("patient_"+ str(i))

    output = pd.DataFrame({"id": patient, 'prediction':y_predicted})
    csv_file = 'prediction.csv'
    output.to_csv(csv_file, index=False)







