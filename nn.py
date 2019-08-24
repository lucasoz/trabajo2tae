import pandas as pd
import seaborn as sns
from math import floor, ceil
from pylab import rcParams

# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt

print(tf.__version__)

sns.set(style='ticks', palette='Spectral', font_scale=1.5)

material_palette = ["#4CAF50", "#2196F3", "#9E9E9E", "#FF9800", "#607D8B", "#9C27B0"]
sns.set_palette(material_palette)
rcParams['figure.figsize'] = 16, 8

plt.xkcd();
random_state = 42
np.random.seed(random_state)
tf.set_random_seed(random_state)

bank_df = pd.read_csv("data/bank-additional.csv", sep=";")

# print(bank_df.shape)
# print(bank_df.columns)

# Los 10 atributos elejidos son:
# 1. age (numérico): Edad del cliente.
# 2. job (categorico): Tipo de trabajo.
# 3. marital (categórico): Estado civil.
# 4. education (categórico): Grado de escolaridad alcanzado.
# 5. default (categórico): El cliente tiene deudas vencidas.
# 6. housing (categórico): El cliente tiene un prestamo de vivienda.
# 7. loan (categórico): El cliente tiene prestamos personales.
# 8. duration (numérico): Duración del contacto (Llamada).
# 9. month (categórico): Mes en el que se realizó el último contacto (Llamada).
# 10. previous (numérico): Número de contactos previos con el cliente.

merge_vector = ["age","job","marital","education", "default","housing","loan","duration", "month","previous","y"]
# print(merge_vector)


duplicated_mask = bank_df.duplicated(keep=False, subset=merge_vector)
duplicated_df = bank_df[duplicated_mask]
unique_df = bank_df[~duplicated_mask]
unique_df = unique_df[merge_vector]

unique_df = unique_df.sample(frac=1)
# Se encontraron 2 datos repetidos

# print(unique_df.shape)
# print(unique_df.columns)
# Descomentar para sacar las gráficas

# unique_df.age.value_counts(bins=10).plot(kind="bar", rot=0);
# plt.show()
#
# unique_df.job.value_counts().plot(kind="bar", rot=0);
# plt.show()
#
# unique_df.marital.value_counts().plot(kind="bar", rot=0);
# plt.show()
#
# unique_df.education.value_counts().plot(kind="bar", rot=0);
# plt.show()
#
# unique_df.default.value_counts().plot(kind="bar", rot=0);
# plt.show()
#
# unique_df.housing.value_counts().plot(kind="bar", rot=0);
# plt.show()
#
# unique_df.loan.value_counts().plot(kind="bar", rot=0);
# plt.show()
#
# unique_df.duration.value_counts(bins=5).plot(kind="bar", rot=0);
# plt.show()
#
# unique_df.month.value_counts().plot(kind="bar", rot=0);
# plt.show()
#
# unique_df.previous.value_counts().plot(kind="bar", rot=0);

# sns.pairplot(unique_df[["age","job","marital","education", "default","housing","loan","duration", "month","previous", "y"]], hue="y");
# plt.show()

# corr_mat = unique_df.corr()
# fig, ax = plt.subplots(figsize=(20, 12))
# sns.heatmap(corr_mat, vmax=1.0, square=True, ax=ax);
# plt.show()

# print(unique_df)

def encode(series):
  return pd.get_dummies(series.astype(str))

# como las variables categóricas tienen unknown en común entonces pueden haber errores a la hora de codificar
# estas variables

unique_df.job = unique_df.job.map('job_{}'.format)
unique_df.marital = unique_df.marital.map('marital_{}'.format)
unique_df.education = unique_df.education.map('education_{}'.format)
unique_df.default = unique_df.default.map('default_{}'.format)
unique_df.housing = unique_df.housing.map('housing_{}'.format)
unique_df.loan = unique_df.loan.map('loan_{}'.format)

train_x = pd.get_dummies(unique_df.job)
train_x['age'] = unique_df.age
train_x['duration'] = unique_df.duration
train_x['previous'] = unique_df.previous
train_x = pd.concat([train_x,
                     encode(unique_df.marital),
                     encode(unique_df.education),
                     encode(unique_df.default),
                     encode(unique_df.housing),
                     encode(unique_df.loan),
                     encode(unique_df.month)], axis = 1)
train_x['jan'] = 0
train_x['feb'] = 0

job = unique_df.job.unique()
marital = unique_df.marital.unique()
education = unique_df.education.unique()
default = unique_df.default.unique()
housing = unique_df.housing.unique()
loan = unique_df.loan.unique()
month = ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec']
variables = np.concatenate((job, ['age', 'duration', 'previous'], marital, education, default, housing, loan, month))
x = train_x[variables]
y = unique_df.y.replace('yes',1).replace('no', 0)

train_size = 0.7

train_cnt = floor(train_x.shape[0] * train_size)
x_train = x.iloc[0:train_cnt].values
y_train = y.iloc[0:train_cnt].values
x_test = x.iloc[train_cnt:].values
y_test = y.iloc[train_cnt:].values

# define the keras model
model = keras.models.Sequential()
model.add(keras.layers.Dense(12, input_dim=48, activation='relu'))
model.add(keras.layers.Dense(48, activation='relu'))
model.add(keras.layers.Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=150, batch_size=10)

# evaluate the keras model
_, accuracy = model.evaluate(x_test, y_test)
print('Accuracy: %.2f' % (accuracy*100))

