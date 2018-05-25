#import matplotlib.pyplot as plt
import numpy.linalg as la
import numpy as np
import math
import csv 

class PCA (object): 
    def __init__(self):
        self.data = []                 # ARMAZENA TODOS OS DADOS SEM A CLASSE 
        self.target = []               # CLASSES DE CADA INSTANCIA 
        self.all_data = []             # INSTANCIAS COM SUAS RESPECTIVAS CLASSES 
        self.normalized_data = []      # INSTANCIAS SUBTRAIDAS DAS MEDIAS
    
    def load(self, filename):
        aux = []
        with open(filename, 'r') as dataset: 
            data = list(csv.reader(dataset))
            for inst in data: 
                inst_class = inst.pop(-1)
                self.target.append(inst_class)
                row = [float(x) for x in inst]
                aux.append(row)
        self.data = np.asarray(aux)
    
    # Extrai as medias dos atributos 
    def mean(self):
        all_data = []
        mean_vec = []
        size_data = len(self.data)
        for i in range(len(self.data[0])):
            row = []
            for j in range(len(self.data)): 
                aux = self.data[j]
                row.append(aux[i])
            all_data.append(row)
        for x in all_data: 
            aux = sum(x)/size_data
            mean_vec.append(aux)

        return mean_vec

    #subtrai a media de toda a base de valores 
    def normalize(self): 
        mean_vector = self.mean()
        aux = [] 
        for x in self.data:
            row = [x[i] - mean_vector[i] for i in range(len(x))]
            aux.append(row)
        self.normalized_data = np.asarray(aux)

    #Retorna a matriz de covariancia 
    def get_covanciance_matrix(self): 
        aux = self.data 
        return np.cov(np.transpose(aux))
    
    #retornar os autovalores e autovetores ordenado 
    def get_values(self): 
        cov = self.get_covanciance_matrix()
        eigenvalue, eigenvector = la.eig(cov)
        ind=np.argsort(eigenvalue)[::-1] 
        eigenvalue_dec = eigenvalue[ind]
        eigenvector_dec = np.transpose(eigenvector)
        eigenvector_dec = eigenvector_dec[ind]
        return (eigenvalue_dec, eigenvector_dec)
    
    def get_evr(self, eigenvalue): 
        return eigenvalue/np.sum(eigenvalue)

    def hotteling_trans(self, comp_number):
        _, eigenvector = self.get_values()
        eigenvector = eigenvector[:comp_number]
        trans = []
        for x in self.data: 
            trans.append(np.dot(eigenvector, x))
        return trans
    
    # Retorna os dados para verificar o desenpenho 
    def get_data(self, filename, comp_number):
        pca = PCA ()
        pca.load('dataset1-1.csv')        
        pca.hotteling_trans(3)

  
