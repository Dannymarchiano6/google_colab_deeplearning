
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

from google.colab import files
uploaded = files.upload()


df = pd.read_csv(next(iter(uploaded)))
df.head()

df = df.dropna()  # Hapus baris kosong

# Label encoding kolom kategorikal (ubah sesuai dataset kamu)
for col in df.select_dtypes(include='object').columns:
    df[col] = LabelEncoder().fit_transform(df[col])

# Misal label kolom 'Label' = 1 (attack), 0 (normal)
X = df.drop('Label', axis=1)
y = df['Label']

# Normalisasi
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# 🤖 5. Model Deep Learning
model = Sequential([
    Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    Dense(32, activation='relu'),
    Dense(1, activation='sigmoid')  # Output biner: 0/1
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# 🧠 6. Training
history = model.fit(X_train, y_train, epochs=20, batch_size=32, validation_split=0.2)

# 📈 7. Evaluasi
loss, acc = model.evaluate(X_test, y_test)
print(f"Akurasi: {acc:.2f}")

y_pred = (model.predict(X_test) > 0.5).astype(int)
print(classification_report(y_test, y_pred))

# 🔧 8. Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()
