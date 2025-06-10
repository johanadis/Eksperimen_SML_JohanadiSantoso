import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler


def preprocess_data(input_path, output_path):
    """
    Fungsi untuk memproses dataset secara otomatis.
    Input: Path ke file dataset mentah (CSV).
    Output: Dataset yang sudah diproses disimpan ke output_path.
    """
    # Memuat dataset
    df = pd.read_csv(input_path)

    # Definisikan kolom kategorikal dan numerik
    categorical_col = ['Stage_fear', 'Drained_after_socializing']
    numerical_col = ['Time_spent_Alone', 'Social_event_attendance',
                     'Going_outside', 'Friends_circle_size', 'Post_frequency']

    # Menangani missing values
    # Untuk fitur numerik: impute dengan median
    numeric_imputer = SimpleImputer(strategy='median')
    df[numerical_col] = numeric_imputer.fit_transform(df[numerical_col])

    # Untuk fitur kategorikal: impute dengan modus
    categorical_imputer = SimpleImputer(strategy='most_frequent')
    df[categorical_col] = categorical_imputer.fit_transform(
        df[categorical_col])

    # Encoding data
    # Encode target variable Personality: Extrovert=0, Introvert=1
    df['Personality'] = (df['Personality'] == 'Introvert').astype(int)

    # Encoding biner untuk kolom kategorikal: Yes=1, No=0
    df[categorical_col] = (df[categorical_col] == 'Yes').astype(int)

    # Normalisasi fitur numerik
    scaler = StandardScaler()
    df[numerical_col] = scaler.fit_transform(df[numerical_col])

    # Menyimpan dataset
    df.to_csv(output_path, index=False)
    print(f"Dataset yang sudah diproses disimpan sebagai {output_path}")

    return df


if __name__ == "__main__":
    input_path = 'personality_dataset.csv'
    output_path = 'preprocessing/personality_dataset_preprocessing.csv'
    preprocess_data(input_path, output_path)
