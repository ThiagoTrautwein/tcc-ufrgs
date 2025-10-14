import pandas as pd

def generate_muse_v017_csv(filepath):
    n_pixels = 784
    n_samples_per_channel = 512
    
    col_names = (
        ["dataset", "origin", "digit_event"] +
        [f"pixel_{i}" for i in range(n_pixels)] +
        ["timestamp"] +
        [f"EEGdata_TP9_{i}" for i in range(n_samples_per_channel)] +
        [f"EEGdata_AF7_{i}" for i in range(n_samples_per_channel)] +
        [f"EEGdata_AF8_{i}" for i in range(n_samples_per_channel)] +
        [f"EEGdata_TP10_{i}" for i in range(n_samples_per_channel)] +
        [f"PPGdata_PPG1_{i}" for i in range(n_samples_per_channel)] +
        [f"PPGdata_PPG2_{i}" for i in range(n_samples_per_channel)] +
        [f"PPGdata_PPG3_{i}" for i in range(n_samples_per_channel)] +
        [f"Accdata_ACCX_{i}" for i in range(n_samples_per_channel)] +
        [f"Accdata_ACCY_{i}" for i in range(n_samples_per_channel)] +
        [f"Accdata_ACCZ_{i}" for i in range(n_samples_per_channel)] +
        [f"Gyrodata_GYRX_{i}" for i in range(n_samples_per_channel)] +
        [f"Gyrodata_GYRY_{i}" for i in range(n_samples_per_channel)] +
        [f"Gyrodata_GYRZ_{i}" for i in range(n_samples_per_channel)]
    )
    
    df = pd.read_csv(filepath, header=None, names=col_names)
    
    return df

def generate_muse_v1_csv():

    col_names = [
        "id", "event", "device", "channel", "code", "size", "data"
    ]

    df = pd.read_csv('datalake/raw/Muse-v1.0/MU.txt', header=None, sep='\t', names=col_names)

    df.to_csv('datalake/processed/Muse-v1.0/Muse-v1.0.csv', index=False, sep=';')

if __name__ == "__main__":
    generate_muse_v1_csv()
