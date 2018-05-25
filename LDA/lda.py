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
                aux.update({x[-1]: [x[:len(x)-1]]})
            else:
                aux[x[-1]].append(x[:len(x)-1])
        self.data_per_class = aux
    
    def mean (self):
        mean_vector = []
        size_data = len(self.data)
        for x in range(len(self.data[0])- 1 ): 
            aux = 0 
            for y in self.all_data: 
                aux += y[x]
            mean_vector.append(aux/size_data)
        return mean_vector

    def mean_per_class(self):
        mean_vector_per_class = {}
        for x in self.data_per_class: 
            data = self.data_per_class[x] 
            size_data = len(data)
            for i in range(len(data[0])): 
                aux = 0
                for y in data: 
                    aux += y[i]
                aux = aux/size_data
                if(x not in mean_vector_per_class.keys()): 
                    mean_vector_per_class.update({x:[aux]})
                else:
                    mean_vector_per_class[x].append(aux)  
        for x in mean_vector_per_class: 
            aux = mean_vector_per_class[x]
            mean_vector_per_class[x] = np.array(aux) 
        return mean_vector_per_class
        
    def variance(self):
        size = len(self.data[0]) 
        #print(size)
        mean_vector = self.mean_per_class()
        
        mean = np.array(self.mean())
        sw = np.zeros((size-1, size-1)) 
        sc = np.zeros((size-1, size-1))  #matriz de covariancia por classe 
        sb =np.zeros((size-1, size-1))  
        for x in self.target: 
            data_aux = self.data_per_class[x]
            size_data = len(data_aux)
            mean_aux = np.array(mean_vector[x]) 
            a = mean_aux - mean
            b = np.transpose(a)
            c = a*b
            sb += c
            sb = sb * size_data 
            for x in data_aux: 
                y = np.array(x)
                normalized = y - mean_aux
                print(len(normalized))
                normalized = normalized.reshape(len(normalized), 1)
                mean_aux = mean_aux.reshape(len(mean_aux), 1)
                sc += normalized* np.transpose(normalized)
            sw += sc
        return sw, sb


        
def main():
    lda = LDA()
    lda.load('dataset1-1.csv')
    lda.divide_per_class()
    #print(lda.mean_per_class())
    lda.variance()
    #print(lda.data_per_class)
    #for x in lda.mean_per_class():
    #    print(x)

if __name__ == '__main__':
	main()