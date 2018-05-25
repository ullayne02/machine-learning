from sklearn.model_selection import StratifiedKFold 
import matplotlib.pyplot as plt
import unicodedata
import operator
import random
import math
import time
import csv

class Knn(object):
    def __init__(self, k):
        self.traning_set = []
        self.testing_set = []
        self.target = set()
        self.preprocessing_set = []
        self.k = k 

    def set_k(self, k): 
        self.k = k
   
    def pre_processor(self):
        for traning_set in self.traning_set: 
            train = []
            for i in range(len(traning_set[0])-1):
                alt = {} 
                for x in traning_set:
                    if x[i] not in alt.keys(): 
                        alt.update({x[i]:1})
                    else:
                        alt[x[i]] += 1

                    if str(x[i])+str(x[-1]) not in alt.keys(): 
                        alt.update({str(x[i])+str(x[-1]):1})
                    else:
                        alt[str(x[i])+str(x[-1])] += 1
                train.append(alt)
            self.preprocessing_set.append(train)     
   
    def get_dist(self, inst1, inst2, k): 
        pass 

    def is_number(self, s):
        try:
            float(s)
            return True
        except ValueError:
            pass
    
        try:
            unicodedata.numeric(s)
            return True
        except (TypeError, ValueError):
            pass
        return False

    def load(self, filename):
        with open(filename, 'r') as dataset: 
            data = csv.reader(dataset)
            data = list(data)
            targ = []
            dataset = []
            all_data = []
            for i in range(len(data)): 
                data[i].reverse() #deixar apenas quando utilizar a base balace sclae
                aux = data[i]
                row = []
                for i in range (len(aux)-1):
                    if(self.is_number(aux[i])): 
                        row.append(float(aux[i]))
                    else: 
                        row.append(aux[i])
                targ.append(aux[-1])
                dataset.append(row)
                row.append(aux[-1])
                all_data.append(row)
        return dataset, targ, all_data
        
    def split_kcross(self, dataset, targ, all_data): 
        skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=None)
        for x in targ: 
            self.target.add(x)
        for x, y in skf.split(dataset, targ): 
            aux_traning = [] 
            aux_testing = []
            aux_testing = [all_data[t] for t in y]
            aux_traning = [all_data[t] for t in x]
            self.traning_set.append(aux_traning)
            self.testing_set.append(aux_testing)

    def get_near_neighboors(self, test_inst1, traning_set, k): 
        near = []
        neighboors = []
        for traning_inst in traning_set:
            near.append((traning_inst, self.get_dist(test_inst1, traning_inst, k)))
        near.sort(key=lambda x:x[1])
        neighboors = near[:self.k]
        '''
        for x in range(self.k):
            neighboors.append(near[x])
        '''
        return neighboors

    def get_response(self, neighboors): 
        votes = {}
        neighboors = [x[0] for x in neighboors]
        for x in neighboors:
            response = x[-1]
            if response not in votes.keys(): 
                votes[response] = 1
            else: 
                votes[response] += 1
        sort = sorted(votes.items(), key=operator.itemgetter(1), reverse=True)
        return sort[0][0]

    def response_weighted(self, neighboors): 
        values = {}

        for x in self.target:
            values[x] = []
        for x in neighboors: 
            values[x[0][-1]].append(x)
        for x in self.target:
            dist = 0
            for y in values[x]:
                if(y[1] !=0): dist += 1/y[1]**2
                else: dist += 0
            values[x] = dist
        sort = sorted(values.items(), key=operator.itemgetter(1), reverse=True)
        return sort[0][0]
    
    def get_accuracy(self, testing_set, predictions):
        c = 0 
        for i in range(len(testing_set)): 
            if(testing_set[i][-1] ==  predictions[i]):
                c += 1
        return c/float(len(testing_set))*100.0 

    def get_accuracy_kcros(self, acuraccy):
        return sum(acuraccy)/10
    
    def show(self, values, time): 
        plt.plot(values, time)
        plt.ylabel('Acuraccy')
        plt.xlabel('Processing Time')
        plt.show()

    def get_acuraccy_by_neighbor(self, dataset, target, all_data):
        # ASSERTS
        k = [1, 3, 5] 
        values = []
        self.split_kcross(dataset, target, all_data)
        for x in k: 
            self.set_k(x)
            acuraccy = []
            test_set = []
            train_set = []
            for i in range(len(self.testing_set)):
                prediction = []
                test_set = self.testing_set[i]
                train_set = self.traning_set[i]            
                for inst in test_set: 
                    neighboors = self.get_near_neighboors(inst, train_set, i)
                    response = self.get_response(neighboors)
                    prediction.append(response) 
                a = self.get_accuracy(test_set, prediction)
                acuraccy.append(a)
            b = self.get_accuracy_kcros(acuraccy)
            print(k, b)
            values.append(b)
        return (k, values) 
class Knn_numeric(Knn): 
    def __init__(self, k): 
        super().__init__(k) 

    def get_dist(self, inst1, inst2, k): 
        return self.get_euclidian_dist(inst1, inst2)

    def get_euclidian_dist(self, inst1, inst2): 
        dist = 0 
        mini = min(inst1[:len(inst1)-1])
        maxi = max(inst1[:len(inst1)-1])
        rang = maxi - mini
        for i in range(len(inst1)-1):
            if(rang != 0):
                dist += math.pow(((inst1[i] - inst2[i])/rang),2)
            else: dist = 0
        return math.sqrt(dist)

class Knn_categorical(Knn):
    def __init__(self, k): 
        super().__init__(k)
    
    def get_dist(self, inst1, inst2, k): 
        return self.get_vdm_dist(inst1, inst2, k) 

    def vdmi(self, a, b, i, k): 
        vdmi = []
        for c in self.target: 
            nia = 0
            nib = 0
            nibc = 0
            niac = 0
            if a not in self.preprocessing_set[k][i].keys(): 
                nia = 0
            else: 
                nia = self.preprocessing_set[k][i][a]
            if (str(a)+str(c)) not in self.preprocessing_set[k][i].keys():
                niac = 0
            else:
                niac = self.preprocessing_set[k][i][str(a)+str(c)]
            if b not in self.preprocessing_set[k][i].keys():
                nib = 0
            else:
                nib = self.preprocessing_set[k][i][b]
            if(str(b) + str(c) not in self.preprocessing_set[k][i].keys()):
                nibc = 0
            else: 
                nibc = self.preprocessing_set[k][i][str(b)+str(c)]
            aux1 = 0
            aux2 = 0
            if(nia != 0): 
                aux1 = niac/nia
            else: 
                aux1 = 0
            if (nib != 0):
                aux2 =  nibc/nib
            else: 
                aux2 = 0
            vdmi.append(math.pow(abs(aux1 - aux2), 2))
        parc_result = sum(vdmi)
        return parc_result
    
    def get_vdm_dist (self, inst1, inst2, k): 
        vdm = []
        for i in range(len(inst1)-1): 
            parc_result = self.vdmi(inst1[i], inst2[i], i, k)
            vdm.append(parc_result)
        result = sum(vdm)
        return math.sqrt(result)

class Knn_misc(Knn_categorical, Knn_numeric):
    def __init__(self, k): 
        super().__init__(k)

    def get_dist(self, inst1, inst2, k): 
        return self.get_hvdm_dist(inst1, inst2, k)
    def get_hvdm_dist(self, inst1, inst2, k):
        dist = 0
        for i in range(len(inst1)-1): 
            if(self.is_number(inst1[i])): 
                dist += self.get_euclidian_disti(inst1[i],inst2[i]) 
            else: 
                dist += self.vdmi(inst1[i], inst2[i], i, k)
        return math.sqrt(dist)
    def get_euclidian_disti(self, a, b): 
        return math.sqrt((a-b)**2)


def main():
    values = []
    
    k = [1, 3, 5]
    q = 2 #Numero da questao
    knn = None 
    if(q == 1): 
        knn = Knn_numeric(0)
    elif(q == 2): 
        knn = Knn_categorical(0)
    else: 
        knn = Knn_misc(0)
    
    dataset, target, all_data = knn.load('dataset2-2.csv')
    knn.split_kcross(dataset, target, all_data)
    knn.pre_processor()
    for x in k: 
        knn.set_k(x)
        acuraccy = []
        test_set = []
        train_set = []
        for i in range(len(knn.testing_set)):
            prediction = []
            test_set = knn.testing_set[i]
            train_set = knn.traning_set[i]            
            for inst in test_set: 
                neighboors = knn.get_near_neighboors(inst, train_set, i)
                response = knn.get_response(neighboors)
                prediction.append(response) 
            a = knn.get_accuracy(test_set, prediction)
            acuraccy.append(a)
        b = knn.get_accuracy_kcros(acuraccy)
        print(k, b)
        values.append(b)

if __name__ == '__main__':
	main()



   