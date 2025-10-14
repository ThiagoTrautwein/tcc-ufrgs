from scipy.signal import butter, filtfilt, iirnotch
import numpy as np
import pandas as pd
import pywt
import json
import os

class Preprocessing:

    def butterworth_highpass(self, data, cutoff, fs, order):
        b, a = butter(order, cutoff / (fs / 2), btype="high", analog=False)
        return filtfilt(b, a, data)

    def notch_filter(self, data, fs, freq, Q):
        b, a = iirnotch(w0=freq/(fs/2), Q=Q)
        return filtfilt(b, a, data)

    def dwt_denoise_reconstruct(self, signal, wavelet='db4', level=3, mode='soft'):
        """
        Decomposição DWT, thresholding em coeficientes de detalhe por nível,
        e reconstrução invertida. Retorna sinal reconstruído (float).
        """
        # decomposição
        coeffs = pywt.wavedec(signal, wavelet=wavelet, level=level)
        # coeffs[0] = cA_n, coeffs[1] = cD_n, ..., coeffs[-1] = cD1

        n = len(signal)
        # threshold e denoise por nível (aplica apenas nos detalhes)
        for i in range(1, len(coeffs)):
            cd = coeffs[i]
            # estimativa robusta do ruído usando MAD do coef. de detalhe do nível i
            sigma = np.median(np.abs(cd)) / 0.6745 if cd.size > 0 else 0.0
            thresh = sigma * np.sqrt(2 * np.log(n)) if sigma > 0 else 0.0
            coeffs[i] = pywt.threshold(cd, thresh, mode=mode)

        # reconstrução inversa
        rec = pywt.waverec(coeffs, wavelet=wavelet)
        # garantir mesmo comprimento (waverec pode retornar ligeiramente maior)
        rec = np.asarray(rec[:n])
        return rec

    def read_input(self, input_path:str) -> pd.DataFrame:
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"Arquivo não encontrado: {input_path}")

        if input_path.endswith('.txt'):
            col_names = ["id", "event", "device", "channel", "code", "size", "data"]

            return pd.read_csv(input_path, header=None, sep='\t', names=col_names)
    
        return pd.read_csv(input_path, sep=';')

    def execute(self, input_path, output_path):
        df = self.read_input(input_path)

        df = df[df['code'] != -1]

        df_other = df[["event", "device", "code", "size"]].drop_duplicates(subset=["event"])

        df_pivot = df.pivot(index="event", columns="channel", values="data").reset_index()
        CHANNELS = [col for col in df_pivot.columns if col not in ["event"]]

        df_pivot = df_pivot.merge(df_other, on="event", how="inner")

        fs = 220

        filtered = []
        for _, row in df_pivot.iterrows():
            filtered_row = {
                'event': row['event'],
                'device': row['device'],
                'code': row['code'],
                'size': row['size']
            }
            for channel in CHANNELS:
                data = np.array([int(x) for x in row[channel].split(',')], dtype=float)

                data = self.butterworth_highpass(data=data, cutoff=0.1, fs=fs, order=5)

                data = self.notch_filter(data=data, fs=fs, freq=60.0, Q=30.0)

                data = self.dwt_denoise_reconstruct(signal=data, wavelet='db4', level=3, mode='soft')

                filtered_row[channel] = ','.join(map(lambda v: f"{v:.6f}", data))

            filtered.append(filtered_row)

        df_filtered = pd.DataFrame(filtered)
        df_filtered.to_csv(output_path, index=False, sep=';')
        print("Filtragem + DWT concluídas e CSV salvo em:", output_path)

if __name__ == "__main__":
    Preprocessing().execute(
        input_path='datalake/processed/Muse-v1.0/Muse-v1.0.csv', 
        output_path='datalake/processed/Muse-v1.0/Muse-v1.0_filtered.csv'
    )