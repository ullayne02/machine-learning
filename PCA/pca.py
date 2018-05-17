import matplotlib.pyplot as plt
import numpy.linalg as la
import numpy as np
import math
import csv 

class PCA (object): 
    def __init__(self):
        self.data = []
        self.target = []
        self.all_data = []
        self.normalized_data = []
    
    def load(self, filename):
        aux = []
        with open(filename, 'r') as dataset: 
            data = list(csv.reader(dataset))
            for inst in data: 
                inst_class = inst.pop(-1)
                self.target.append(inst_class)
                row = [float(x) for x in inst]
                aux.append(row)
                #row.append(inst_class)
                #self.all_data.append(row)
        print(aux)
        self.data = np.asarray(aux)
        
    
    def mean(self):
        all_data = []
        mean_vec = []
        size_data = len(self.data)
        for i in range(len(self.data[0]) - 1):
            row = []
            for j in range(len(self.data)): 
                aux = self.data[j]
                row.append(aux[i])
            all_data.append(row)
        for x in all_data: 
            aux = sum(x)/size_data
            mean_vec.append(aux)

        return mean_vec

    def normalize(self): 
        mean_vector = self.mean()
        for x in self.data:
            row = [x[i] - mean_vector[i] for i in range(len(x)-1)]
            self.normalized_data.append(row)
    
    def get_covanciance_matrix(self): 
        return np.cov(np.transpose(self.data))

    #retornar ordenado 
    def get_values(self): 
        cov = self.get_covanciance_matrix()
        eigenvalue, eigenvector = la.eig(cov)
        ind=np.argsort(eigenvalue)[::-1] 
        eigenvalue_dec = eigenvalue[ind]
        eigenvector_dec = eigenvector[ind]
        return (eigenvalue_dec, eigenvector_dec)
    
    def get_evr(self, eigenvalue): 
        return eigenvalue/np.sum(eigenvalue)
    

def main():
    pca = PCA ()
    pca.load('dataset1-1.csv')
    #pca.normalize()
    #y = pca.mean()
    #a = pca.normalized_data
    print('data normalizada:', pca.get_covanciance_matrix())
    eigenvalue, eigenvector = pca.get_values()

    print('autovalor e autovetor:', pca.get_evr(eigenvalue) )
    #print('vetor medio:', y)
    
main()