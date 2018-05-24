import numpy.linalg as la
import numpy as np
import math
import csv 

class LDA(object): 
    def __init__(self):
        self.data = []                 # ARMAZENA TODOS OS DADOS SEM A CLASSE 
        self.target = []               # CLASSES DE CADA INSTANCIA 
        self.all_data = []             # INSTANCIAS COM SUAS RESPECTIVAS CLASSES 
        self.normalized_data = []      # INSTANCIAS SUBTRAIDAS DAS MEDIAS
        self.data_per_class = {}
    
    # carrega os dados do dataset 
    def load(self, filename):
        aux = []
        with open(filename, 'r') as dataset: 
            data = list(csv.reader(dataset))
            row = []
            for inst in data: 
                inst_class = inst.pop(-1)
                self.target.append(inst_class)
                row = [float(x) for x in inst]
                aux.append(row)
                row.append(inst_class)
                self.all_data.append(row)
        self.data = np.asarray(aux)

    # divide do dataset em um dicionario por classe 
    def divide_per_class(self):
        aux = {}
        for x in self.all_data: 
            if(x[-1] not in aux.keys()):
                aux.update({x[-1]: x})
            else:
                aux[x[-1]].append(x)
        self.data_per_class = aux
    
    def mean (self):
        mean_vector = []
        size_data = len(self.data)
        for x in range(len(self.data[0])- 1 ): 
            aux = 0 
            for y in self.all_data: 
                aux += y[x]
            mean_vector.append(aux/size_data)
    
    def()

def main():
    lda = LDA()
    lda.load('dataset1-1.csv')
    lda.divide_per_class()
    print(lda.data_per_class)

if __name__ == '__main__':
	main()