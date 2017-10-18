from __future__ import division

import matplotlib.pyplot as plt

from Data import*
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix

class SVM_tests:

    def tarea3(self,kernel, degree, gamma, minTreshold, maxTreshold,step):

        kernelType = kernel
        #importamos datos
        d = Data("sensorless_tarea2.txt")
        d.filterClasss(48)
        d.run(48)
        #generamos nuestor rango de umbrales e iniciamos variables
        rangeTreshold = np.arange(minTreshold,maxTreshold,step)
        fpr_train_Array = [0] * len(rangeTreshold)
        tpr_train_Array = [0] * len(rangeTreshold)

        fpr_test_Array = [0] * len(rangeTreshold)
        tpr_test_Array = [0] * len(rangeTreshold)

        print "Thinking ..."

        #asignamos conjuntos de prueba y entranamiento
        x_train, y_train = d.baseEntrenamiento, d.entrenamientoClases
        x_test, y_test = d.basePrueba, d.pruebaClases

        #normalizamos
        scaler = StandardScaler().fit(x_train)
        x_train = scaler.transform(x_train)
        scaler = StandardScaler().fit(x_test)
        x_test = scaler.transform(x_test)

        #Numero de 0s (false) y 1s (true), en este caso calculamos el total por clase sin utilizar la matriz de
        #confusion
        false_test = (len(x_test) - sum(y_test))
        false_train = (len(x_train) - sum(y_train))
        true_test = sum(y_test)
        true_train = sum(y_train)

        for i in range(0, len(rangeTreshold)):

            print "...",
            if (kernelType=='linear'):
                clf =SVC(kernel=kernelType)
            if (kernelType=='poly'):
                clf =SVC(kernel=kernelType, degree=degree)
            if (kernelType=='rbf'):
                clf =SVC(kernel=kernelType, gamma=gamma) #por defecto el gamma vale 1/n_features

            #entrenamos la SVN
            clf.fit(x_train,y_train)

            treshold = rangeTreshold[i]

            #predecimos la clase
            score_test = clf.decision_function(x_test)
            pred_test_score = (np.array)([1. if x>treshold else 0. for x in score_test])

            # Calculamos los falsos positivos y verdaderos positivos para el conjunto prueba
            fpr_test = (float)(confusion_matrix(y_test,pred_test_score)[0][1])/false_test
            tpr_test = (float)(confusion_matrix(y_test,pred_test_score)[1][1])/true_test

            #predecimos la clase
            score_train = clf.decision_function(x_train)
            pred_train_score = (np.array)([1. if x>treshold else 0. for x in score_train])

            #Calculamos los falsos positivos y verdaderos positivos para el conjunto entrenamiento
            fpr_train = (confusion_matrix(y_train,pred_train_score)[0][1])/false_train
            tpr_train = (confusion_matrix(y_train,pred_train_score)[1][1])/true_train

            fpr_test_Array[i] = fpr_test
            tpr_test_Array[i] = tpr_test

            fpr_train_Array[i] = fpr_train
            tpr_train_Array[i] = tpr_train
        print "true positive rate sample test"
        print tpr_test_Array
        print "false positive rate sample test"
        print fpr_test_Array

        f = plt.figure(1)

        if (kernelType == 'linear'):
            f.suptitle("Linear")
        if (kernelType == 'poly'):
            f.suptitle("Polynomial, grado : " + str(degree))
        if (kernelType == 'rbf'):
            f.suptitle("RBF, gamma : " +str(gamma))

        #plot ROC curve
        plt.plot([0.,1.],[0.,1.],'k--')
        plt.plot(fpr_test_Array,tpr_test_Array, color='blue')
        plt.plot(fpr_train_Array,tpr_train_Array, color='gray')
        plt.xlabel("FPR")
        plt.ylabel("TPR")
