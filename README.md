#impor Pustaka yang diperlukan
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

#membuat dataframe
data = {
    'Tipe Motor': ['Revo 110 FI', 'Blade 125 FI', 'Supra X 125 FI', 'BeAT eSP', 'BeAT eSP dengan ISS', 'Genio', 'Spacy PGM-FI', 'Scoopy eSP', 'Vario 110 eSP', 'Vario 125 eSP', 'Vario 150 eSP', 'Vario 160 eSP', 'Sonic 150', 'New Verza', 'New Mega Pro FI', 'New CB150 R Streetfire', 'All New CBR150 R', 'PCX 150', 'PCX 160', 'ADV 150', 'ADV 160', 'CB 150X'],
    'Konsumsi BBM per Liter': [62.2, 61.8, 61.8, 58.5, 63, 59.1, 41, 61.9, 59, 59.5, 52.9, 46.9, 40.9, 48, 46.2, 37.87, 41.1, 41, 42.2, 42.2, 45, 45.3]
}
df = pd.DataFrame(data)
print(df)

#membagi data menjadi var independen (X) dan data var independen (y)


X = df.index.values.reshape(-1, 1)
y = df['Konsumsi BBM per Liter']

#membagi data menjadi data latih dan data uji
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#membuat model regresi linear
model = LinearRegression()
model.fit(
    X_train, y_train
)

#membuat prediksi dengan data uji
y_pred = model.predict(X_test)

#menampilkan hasil prediksi
print("Hasil Prediksi: ")
print(pd.DataFrame({'Actual': y_test, 'Predicted' : y_pred}))

#visualisasi hasil prediksi
plt.scatter(X_test, y_test, color='black')
plt.plot(X_test, y_pred, color='blue', linewidth=3)
plt.title('Regresi Linear - Prediksi Konsumsi BBM per Liter')
plt.xlabel('index')
plt.ylabel('Konsumsi BBM per liter')
plt.show()
