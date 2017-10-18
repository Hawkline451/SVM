import numpy as np
from sklearn.model_selection import train_test_split

class Data:
    def __init__(self, fileName ):
        self.db = np.loadtxt(open(fileName, "rb"), delimiter=",", skiprows=0, dtype=None)
        self.clases = []
        self.baseEntrenamiento =np.array([])
        self.entrenamientoClases = ()
        self.basePrueba = np.array([])
        self.pruebaClases = ()

    def run(self,numCaracteristicas):

        np.random.shuffle(self.db)

        x = self.db[:,:numCaracteristicas]
        y = self.db[:,numCaracteristicas]
        #Train Test split valida internamente la representatividad de las clases.
        self.baseEntrenamiento, self.basePrueba, self.entrenamientoClases, self.pruebaClases = train_test_split(
            x , y, test_size=0.2)

    #Filtramos las caracteriticas para moters con o sin fallas
    def filterClasss(self,numCaracteristicas):
        for i in range(0,len(self.db)):
            if (self.db[i][numCaracteristicas]==1):
                self.db[i][numCaracteristicas]=0
            else:
                self.db[i][numCaracteristicas]=1

