import matplotlib.pyplot as plt
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
    
    # Carrega os dados do dataset 
    def load(self, filename):
        aux = []
        with open(filename, 'r') as dataset: 
            data = list(csv.reader(dataset))
            row_all = []
            row = []
            for inst in data: 
                inst_class = inst.pop(-1)
                self.target.append(inst_class)
                row = [float(x) for x in inst]
                row_all =  [float(x) for x in inst]
                row_all.append(inst_class)
                self.all_data.append(row_all)
                aux.append(row)
        self.data = np.asarray(aux)

    # Divide do dataset em um dicionario por classe 
    def divide_per_class(self):
        aux = {}
        for x in self.all_data: 
            if(x[-1] not in aux.keys()):
                aux.update({x[-1]: [x[:len(x)-1]]})
            else:
                aux[x[-1]].append(x[:len(x)-1])
        self.data_per_class = aux
    
    # Retorna o vetor medio do dataset 
    def mean (self):
        mean_vector = []
        size_data = len(self.data)
        for x in range(len(self.data[0])): 
            aux = 0 
            for y in self.all_data: 
                aux += y[x]
            mean_vector.append(aux/size_data)
        return mean_vector

    # Retorna o vetor medio de cada classe 
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
        mean_vector = self.mean_per_class()
        
        mean = np.array(self.mean())
        sw = np.zeros((size, size)) 
        sc = np.zeros((size, size))  #matriz de covariancia por classe 
        sb =np.zeros((size, size))  
        for x in self.target: 
            data_aux = self.data_per_class[x]
            mean_aux = np.array(mean_vector[x]) 
            a = mean_aux - mean
            b = np.transpose(a)
            c = a*b
            sb += c
            for x in data_aux: 
                y = np.array(x)
                y = y.reshape(len(y), 1)
                mean_aux = mean_aux.reshape(len(mean_aux), 1)
                normalized = y - mean_aux
                sc += normalized * np.transpose(normalized)
            sw += sc
        return sw, sb
    
    # Retorna os autovalores e vetores ordenados 
    def get_values(self):
        sw, sb = self.variance()
        inverse = la.inv(sw)
        eig_val, eig_vec = la.eig(inverse.dot(sb))
        eig_pairs = [(np.abs(eig_val[i]), eig_vec[:,i]) for i in range(len(eig_val))]
        eig_pairs = sorted(eig_pairs, key=lambda k: k[0], reverse=True)
        return eig_pairs

    # Transformacao para diminuir a dimensao do dataset 
    def transf (self, comp_number): 
        eig_pairs = self.get_values()
        eig_vec = [b for a, b in eig_pairs]
        eig_vec = np.array(eig_vec[:comp_number])
        for x in eig_vec:
            x.reshape(len(eig_vec[0]), 1)
        return np.dot(self.data, eig_vec.real.T)

    # Retorna os dados para verificar o desempenho 
    def get_data(self, filename, comp_number):
        self.load(filename)
        self.divide_per_class()
        a = list(self.transf(comp_number))
        all_data = []
        for i in range(len(a)):
            aux = list(a[i])
            aux.append(self.target[i])
            all_data.append(aux)
        return (a, self.target, all_data)
