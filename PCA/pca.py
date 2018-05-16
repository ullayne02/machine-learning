import matplotlib.pyplot as plt
import random
import math
import csv 

class PCA (object): 
    def __init__(self):
        self.data = []
    
    def load(self, filename):
        with open(filename, 'r') as dataset: 
            data = list(csv.reader(dataset))
            for inst in data: 
                inst_class = inst.pop(-1)
                row = [float(x) for x in inst]
                row.append(inst_class)
                self.data.append(row)
    
    def mean_vector(self, instance):
        all_data = []
        mean_vec = []
        for i in range(len(self.data[0]) - 1):
            row = []
            for j in range(len(data)): 
                aux = data[j]
                row.append(aux[i])
            all_data.append(row)
        
        for x in all_data: 
            aux = sum(x)/len(self.data)
            mean_vec.append(aux)

        return mean_vec
    
    #tirar duvida se o a media e da classe ou da instancia:  DONE 
    def normalize(self, inst): 
        inst_class = inst.pop(-1)
        mean = self.mean(inst)
        aux = [x-mean for x in inst]
        aux.append(inst_class)
        return aux
    
    def variance(self, inst1, inst2): 
        inst_class1 = inst1.pop(-1)
        inst_class2 = inst2.pop(-1)
        mean1 = self.inst1
        mean2 = self.inst2

    
    def covanciance_matrix(self): 
        for x in self.data: 
            for y in self.data: 

def main():
    pca = PCA ()
    pca.load('dataset1-1.csv')
    i = 1
    for x in pca.data:
        y = pca.normalize(x)
        print('normal', x)
        print('normalized:', y)
        print('-----')
        i+= 1
        if(i == 10): break

main()