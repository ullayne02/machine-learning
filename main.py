from KNN import Knn 
from LDA import lda 
from PCA import pca 

def main(): 
    #ASSERTS
    knn_numeric = Knn.Knn_numeric(0)
    filename = 'data/dataset1-1.csv'
    comp_number = 1            #Numero de componentes principais

    #PCA
    #pca_inst = pca.PCA()
    #(data, target, all_data) = pca_inst.get_data(filename, comp_number)
    
    #LDA 
    lda_inst = lda.LDA()
    (data, target, all_data) = lda_inst.get_data(filename, comp_number)
    (k, values) = knn_numeric.get_acuraccy_by_neighbor(data, target, all_data)    

    print(list(zip(k, values)))

if __name__ == '__main__':
	main()
