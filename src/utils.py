import pandas as pd

def load_data(DATA_PATH, DATASET_NAME):
    col_names = (
            ['unit_number', 'cycle', 'setting1', 'setting2', 'setting3']
            + [f'sensor{i}' for i in range(1, 22)]
        )

    train_df = pd.read_csv(
            f'{DATA_PATH}/train_{DATASET_NAME}.txt', sep=r'\s+', header=None, names=col_names
        )
    test_df = pd.read_csv(
            f'{DATA_PATH}/test_{DATASET_NAME}.txt', sep=r'\s+', header=None, names=col_names
        )
    act_df = pd.read_csv(
            f'{DATA_PATH}/RUL_{DATASET_NAME}.txt', sep=r'\s+', header=None, names=['RUL']
        )
    return train_df, test_df, act_df

def prepare_data(df, threshold=125):
        df = df.copy()
        df['max_cycle'] = df.groupby('unit_number')['cycle'].transform('max')
        df['RUL'] = df['max_cycle'] - df['cycle']
        df['RUL'] = df['RUL'].clip(upper=threshold)

        X = df.drop(columns=['RUL', 'max_cycle'])
        y = df['RUL']

        return X, y