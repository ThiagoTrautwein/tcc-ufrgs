import os 
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Bidirectional, LSTM, Dropout, Flatten, Dense, ELU
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    tf.config.experimental.set_virtual_device_configuration(
        gpus[0],
        [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=4000)]
    )

def ler_csv():
    CHANNELS = ['FP1', 'FP2', 'TP10', 'TP9']
    TARGET_LEN = 440

    df = pd.read_csv('datalake/processed/Muse-v1.0/Muse-v1.0_filtered.csv', sep=';')
    df = df[df['code'] != -1]

    X_list = []
    y_list = []

    for _, row in df.iterrows():
        channels_data = []

        for ch in CHANNELS:
            arr = np.array([float(x) for x in row[ch].split(',')])
            if len(arr) > TARGET_LEN:
                arr = arr[:TARGET_LEN]
            elif len(arr) < TARGET_LEN:
                arr = np.pad(arr, (0, TARGET_LEN - len(arr)), mode='constant')
            channels_data.append(arr)

        # Empilha canais → shape (TARGET_LEN, n_canais)
        sample = np.stack(channels_data, axis=1)
        X_list.append(sample)

        y_list.append(int(row['code']))

    # Empilha canais → shape (n_amostras, TARGET_LEN, n_canais)
    X = np.stack(X_list, axis=0)
    y = np.array(y_list, dtype=int)

    return X, y

def create_model():
    N_CHANNELS = 4
    N_SAMPLES = 440

    model = Sequential([
        Input(shape=(N_SAMPLES, N_CHANNELS)),
        Bidirectional(LSTM(N_SAMPLES, return_sequences=True)),
        Dropout(0.1),
        Bidirectional(LSTM(N_SAMPLES // 2, return_sequences=True)),
        Dropout(0.1),
        Bidirectional(LSTM(N_SAMPLES // 4, return_sequences=True)),
        Dropout(0.1),
        Flatten(),
        Dense(128, activation='elu'),
        Dense(10, activation='softmax')
    ])

    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    return model

def train(X, y):
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)

    print(f"Treino: {X_train.shape}, Validação: {X_val.shape}, Teste: {X_test.shape}")

    early_stop = EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True
    )

    checkpoint = ModelCheckpoint(
        'melhor_modelo.h5',
        monitor='val_accuracy',
        save_best_only=True,
        mode='max',
        verbose=1
    )

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=10, 
        batch_size=256,
        callbacks=[early_stop, checkpoint],
        verbose=1
    )

def validate(model, X_test, y_test):
    y_pred_probs = model.predict(X_test)
    y_pred = np.argmax(y_pred_probs, axis=1)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average='macro')
    rec = recall_score(y_test, y_pred, average='macro')
    f1 = f1_score(y_test, y_pred, average='macro')

    print(f"\n📊 Desempenho no conjunto de teste:")
    print(f"Acurácia: {acc:.4f}")
    print(f"Precisão (macro): {prec:.4f}")
    print(f"Recall (macro): {rec:.4f}")
    print(f"F1-score (macro): {f1:.4f}")

    print("\nRelatório por classe:")
    print(classification_report(y_test, y_pred, digits=4))

def normalize(X_train:np.ndarray, X_val:np.ndarray, X_test:np.ndarray):

    mu = X_train.mean(axis=(0, 1), keepdims=True)
    sigma = X_train.std(axis=(0, 1), keepdims=True)

    # Evitar divisão por zero
    sigma[sigma == 0] = 1.0

    X_train_z = (X_train - mu) / sigma
    X_val_z   = (X_val - mu) / sigma
    X_test_z  = (X_test - mu) / sigma

    n_train, n_points, n_channels = X_train_z.shape

    # Achatar apenas amostras e tempo (mantendo canais separados)
    X_train_2d = X_train_z.reshape(-1, n_channels)
    X_val_2d   = X_val_z.reshape(-1, n_channels)
    X_test_2d  = X_test_z.reshape(-1, n_channels)

    scaler = MinMaxScaler(feature_range=(0, 1))
    X_train_scaled = scaler.fit_transform(X_train_2d)
    X_val_scaled   = scaler.transform(X_val_2d)
    X_test_scaled  = scaler.transform(X_test_2d)

    X_train_final = X_train_scaled.reshape(n_train, n_points, n_channels)
    X_val_final   = X_val_scaled.reshape(X_val.shape[0], n_points, n_channels)
    X_test_final  = X_test_scaled.reshape(X_test.shape[0], n_points, n_channels)

    return X_train_final, X_val_final, X_test_final

if __name__ == "__main__":

    model = create_model()
    X, y = ler_csv()

    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)

    X_train, X_val, X_test = normalize(X_train, X_val, X_test)

    print(f"Treino: {X_train.shape}, Validação: {X_val.shape}, Teste: {X_test.shape}")

    early_stop = EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True
    )

    checkpoint = ModelCheckpoint(
        'melhor_modelo.keras',
        monitor='val_accuracy',
        save_best_only=True,
        mode='max',
        verbose=0
    )

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=10, 
        batch_size=16,
        callbacks=[early_stop, checkpoint],
        verbose=1
    )

    validate(model, X_test, y_test)
