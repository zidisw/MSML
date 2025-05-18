import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
import sys

def preprocess_data(input_path, output_path):
    # Memuat dataset
    df = pd.read_csv(input_path)
    
    # Penanganan nilai negatif
    df['SO2'] = df['SO2'].clip(lower=0)
    df['PM10'] = df['PM10'].clip(lower=0)
    
    # Penanganan outlier
    def remove_outliers(df, column):
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        df = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
        return df
    
    df = remove_outliers(df, 'PM2.5')
    df = remove_outliers(df, 'PM10')
    
    # Encoding Air Quality
    label_encoder = LabelEncoder()
    df['Air Quality'] = label_encoder.fit_transform(df['Air Quality'])
    
    # Normalisasi fitur numerik
    numerical_cols = ['Temperature', 'Humidity', 'PM2.5', 'PM10', 'NO2', 'SO2', 'CO', 'Proximity_to_Industrial_Areas', 'Population_Density']
    scaler = StandardScaler()
    df[numerical_cols] = scaler.fit_transform(df[numerical_cols])
    
    # Menyimpan dataset yang telah diproses
    df.to_csv(output_path, index=False)
    print(f'Dataset yang telah diproses disimpan sebagai {output_path}')
    return df

if __name__ == '__main__':
    # Ambil path dari command line argument
    input_path = sys.argv[1]
    output_path = sys.argv[2]
    preprocess_data(input_path, output_path)