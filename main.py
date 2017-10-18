from __future__ import division
from SVM_tests import*

#
#Aca cambiamos nuestro kernel a eleccion apra realizar las diferentes pruebas
#kernel : 'poly','rbf','linear'
#tarea3(kernel, degree, gamma, minTreshold, maxTreshold,step), por defecto el umbral es 0 osea clasifica segun su signo, positivo o negativo
#Si el kernel no posee un atributo usar 0 o simplemente anotar cualquier numero, para ciertos kernels se ignoran
#ciertos parametros

degree = 3
gamma = 1/50

#podemos cambiar parametro y probar los distintos resultados
tests = SVM_tests()
tests.tarea3('linear', degree, gamma, -8, 12, .5)
#tests.tarea3('poly', degree, gamma, -6, 6, .3)
#tests.tarea3('rbf', degree, gamma, -4, 4, .2)
plt.show()