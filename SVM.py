import numpy as np
import matplotlib.pyplot as plt
from Data import*
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix


#Esta es una version basica de prueba que permite revisar el funcionamiento de una SVN
d = Data("sensorless_tarea2.txt")
d.filterClasss(48)
d.run(48)

x_train, y_train = d.baseEntrenamiento, d.entrenamientoClases
x_test, y_test = d.basePrueba, d.pruebaClases

scaler = StandardScaler().fit(x_train,y_train)
x_train = scaler.transform(x_train)
scaler = StandardScaler().fit(x_test)
x_test = scaler.transform(x_test)


clf =SVC(kernel='linear')
clf.fit(x_train,y_train)

#pred_test = clf.predict(x_test)
#pred_train = clf.predict(x_train)

treshold = 0
score_test = clf.decision_function(x_test)
pred_test_score = (np.array)([1. if x>treshold else 0. for x in score_test])

#podemos calcular el total de clases 1 y 0 sumando filas de la matriz de confusion.
fpr_test = (float)(confusion_matrix(y_test,pred_test_score)[0][1])/(sum(confusion_matrix(y_test,pred_test_score)[:,:][0]))
tpr_test = (float)(confusion_matrix(y_test,pred_test_score)[1][1])/(sum(confusion_matrix(y_test,pred_test_score)[:,:][1]))

print clf.score(x_train,y_train)
print clf.score(x_test,y_test)

