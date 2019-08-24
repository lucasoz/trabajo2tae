# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

# Helper libraries
import numpy as np
import pandas as pd
from math import floor

random_state = 42
np.random.seed(random_state)
tf.set_random_seed(random_state)

# https://archive.ics.uci.edu/ml/datasets/Bank+Marketing
bank_df = pd.read_csv("data/bank-additional.csv", sep=";")

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

duplicated_mask = bank_df.duplicated(keep=False, subset=merge_vector)
duplicated_df = bank_df[duplicated_mask]
unique_df = bank_df[~duplicated_mask]
unique_df = unique_df[merge_vector]

unique_df = unique_df.sample(frac=1)
# Se encontraron 2 datos repetidos
3
def encode(series):
  return pd.get_dummies(series.astype(str))

# Como las variables categóricas tienen unknown en común entonces pueden haber errores a la hora de codificar estas variables

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

# This months not appear in data, then set 0 by default
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

# Define the keras model
model = keras.models.Sequential()
model.add(keras.layers.Dense(12, input_dim=48, activation='relu'))
model.add(keras.layers.Dense(48, activation='relu'))
model.add(keras.layers.Dense(24, activation='relu'))
model.add(keras.layers.Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(x_train, y_train, epochs=150, batch_size=10)

# Evaluate the keras model
_, accuracy = model.evaluate(x_test, y_test)
print('Neural Network Accuracy: %.2f' % (accuracy*100))

# Se crea un DataFrame con una fila de 0 por defecto, más adelante según los datos entrados se cambiarán los valores respectivos
x_prediction = pd.DataFrame(np.array([np.zeros(48)]), columns=variables)

# Función que permite Listar opciones y validar la elección de variables categóricas
def optionsCat(lista):
    while True:
        for idx, j in enumerate(lista):
            print("(%d) %s" % (idx, j))
        try:
            responseInt = int(input("Please, enter a number: "))
        except ValueError:
            print("This is not a number.")
        except Exception:
            print("This is not a option.")
        else:
            if(responseInt < 0 or responseInt >= len(lista)):
                print("This option is not valid.")
                pass
            else:
                print('Thanks, %s is selected option' % (lista[responseInt]))
                break

    return lista[responseInt]

# Función que permite Listar opciones y validar la elección de variables numéricas
def optionsNumber():
    while True:
        try:
            responseInt = int(input("Please, enter a number: "))
        except ValueError:
            print("This is not a number.")
        except Exception:
            print("This is not a option.")
        else:
            print('Thanks, %d is selected option' % (responseInt))
            break

    return responseInt

print("Select a job [For prediction OK use admin.]:")
job_option = optionsCat(job)
x_prediction[job_option][0] = 1

print("Select a marital state [For prediction OK use single]:")
marital_option = optionsCat(marital)
x_prediction[marital_option][0] = 1

print("Select a education level [For prediction OK use university.degree]:")
edication_option = optionsCat(education)
x_prediction[edication_option][0] = 1

print("Select a default state [For prediction OK use no]:")
default_option = optionsCat(default)
x_prediction[default_option][0] = 1

print("Select a housing state [For prediction OK use no]:")
housing_option = optionsCat(housing)
x_prediction[housing_option][0] = 1

print("Select a loan state [For prediction OK use no]:")
loan_option = optionsCat(loan)
x_prediction[loan_option][0] = 1

print("Select the last contact month [For prediction OK use oct]:")
month_option = optionsCat(month)
x_prediction[month_option][0] = 1

print("Enter the age of client in years [For prediction OK use 28]:")
age_value = optionsNumber()
x_prediction['age'][0] = age_value

print("Enter the call duration in seconds [For prediction OK use 281]:")
duration_value = optionsNumber()
x_prediction['duration'] = duration_value

print("Enter how many previous contact have been [For prediction OK use 2]:")
previous_value = optionsNumber()
x_prediction['previous'] = previous_value

y_prediction = model.predict_classes(x_prediction)
if y_prediction[0][0] == 1:
    print("The Neural Network predicts that client offert could be successfully")
else:
    print("The Neural Network predicts that client offert could be fail")
