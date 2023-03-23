# -*- coding: utf-8 -*-
from sklearn.datasets import load_digits
digitos = load_digits()



import matplotlib.pyplot as plt
plt.gray()
plt.matshow(digitos.images[234]) 
plt.show() 
print("Clase de la imagen: {}".format(digitos.target[234]))



from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

# Dividir los datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(digitos.data, digitos.target, test_size=0.3, random_state=42)

# Entrenar un clasificador SVM
svm = SVC(kernel='linear')
svm.fit(X_train, y_train)

# Evaluar el desempeño en el conjunto de prueba
score = svm.score(X_test, y_test)
#print("Exactitud del clasificador multiclase: {:.2f}".format(score))

# Seleccionar solo los datos correspondientes a los dígitos 4 y 9
X_bin = digitos.data[(digitos.target == 4) | (digitos.target == 9)]
y_bin = digitos.target[(digitos.target == 4) | (digitos.target == 9)]

# Dividir los datos en entrenamiento y prueba
X_bin_train, X_bin_test, y_bin_train, y_bin_test = train_test_split(X_bin, y_bin, test_size=0.3, random_state=42)

# Entrenar un clasificador SVM binario
svm_bin = SVC(kernel='linear')
svm_bin.fit(X_bin_train, y_bin_train)

# Evaluar el desempeño en el conjunto de prueba
score_bin = svm_bin.score(X_bin_test, y_bin_test)
#print("Exactitud del clasificador binario: {:.2f}".format(score_bin))


# Mostrar un ejemplo de dígito que se clasifique bien como un 4
plt.matshow(X_bin_test[0].reshape(8,8))
plt.title("Predicción correcta: {}".format(svm_bin.predict(X_bin_test[0].reshape(1, -1))[0]))
#plt.show()

# Mostrar un ejemplo de dígito que se clasifique bien como un 9
plt.matshow(X_bin_test[6].reshape(8,8))
plt.title("Predicción correcta: {}".format(svm_bin.predict(X_bin_test[6].reshape(1, -1))[0]))
#plt.show()

# Mostrar un ejemplo de dígito que se clasifique mal
plt.matshow(X_bin_test[2].reshape(8,8))
plt.title("Predicción incorrecta: {}".format(svm_bin.predict(X_bin_test[2].reshape(1, -1))[0]))
#plt.show()
