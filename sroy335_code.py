import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

import pandas as pd
import csv #not needed unless you use ReadCsv()
import time

from sklearn.cluster import KMeans
from sklearn import datasets
from sklearn.utils import shuffle
from sklearn.metrics import silhouette_score
from sklearn.metrics import calinski_harabasz_score
from sklearn.metrics import davies_bouldin_score
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

from sklearn.decomposition import FastICA
from sklearn import random_projection
from sklearn.decomposition import TruncatedSVD


#TODO: 2nd data set
#Learning Curves
#2 hyperparameters to tune
#Cross-validation: scores = cross_val_score(clf, X, y, cv=5)
#Show validation curves

DEFAULT_TEST_SIZE = 0.2



def Wallclock(X,y,dataset=1):
    neu_model = MLPClassifier(random_state=1,hidden_layer_sizes = (50,), activation='logistic' )

    
    #models = [neu_model,dt_model,svm_model,boost_model,neigh_models]
    models = [neu_model]
    model_names = ["Neural_Nets_CV","DT_CV", "SVM_CV","Boosting_CV","KNN_CV"]

    X, y, dataset_name = GetData(dataset)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=0)
    
    i=0
    train_sizes = [0.8]
    for model in models:
        print("\n model: ",model_names[i])
        start = time.time()
        model.fit(X_train,y_train)
        end = time.time()
        print("train clock time in ms: ", (end-start)*1000)

        
        start = time.time()
        model.predict(X_test)
        end = time.time()
        print("test clock time in ms : ", (end-start)*1000)

        train_score = 100*model.score(X_train, y_train)
        test_score = 100*model.score(X_test, y_test)
        print ("train score", train_score)
        print("test score", test_score)
        i+=1



        

def Neural_Net(X,y,dimalgo,dataset=1):
    neu_model = MLPClassifier(random_state=1)
    #plot learning curve
    lrn_crv = run_algo(neu_model,X,y,dataset)
    name = "Neutal_Network_"+dimalgo+"_"
    plot_learning_curve(lrn_crv, name,dataset)
    

    """
    # tune hyperparameter
    hidden_layers = []
    models = []

    for h in range (10,200,10):
        model = MLPClassifier(random_state=1, hidden_layer_sizes=(h,))
        models.append(model)
        hidden_layers.append(h)

    plot_hyperparmeter_performance(models, "hidden layer sizes", np.array(hidden_layers), "Neural_Nets",dataset)


    # tune hyperparameter activation
    activations = ['identity', 'logistic', 'tanh', 'relu']
    models = []

    for a in activations:
        model = MLPClassifier(random_state=1, activation=a)
        models.append(model)
  

    plot_hyperparmeter_performance(models, "activation functions", np.array(activations), "Neural_Nets",dataset)
    """
        

   
    



def run_algo(model,X,y,dataset=1):
    
    test_percs = []
    lrn_crv = []
    
    scale = 4
    for i in range(9*scale, 1, -1):
        test_percs.append(float(i)/(10*scale))
        
    for test_perc in test_percs:
    
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = test_perc, random_state=0)
        train_perc = 100*round(1-test_perc,4)
        
        print("-----\nTraining percentage = ", train_perc)
        print("\n")
        
        
        model.fit(X_train, y_train)
        train_score = 100*model.score(X_train, y_train)
        test_score = 100*model.score(X_test, y_test)

        print("Train score: ",train_score)
        print("Test score: ",test_score)

        lrn_crv.append([train_perc,train_score,test_score])
        
    return lrn_crv


def plot_learning_curve(lrn_crv,algo_name,dataset=1):
    crv = np.array(lrn_crv)
    x_data = crv[:,0] #train_percentages
    y_data = crv[:,1:] # train and test scores
    figure_name = algo_name + "Dataset-" + str(dataset) + ".jpg"
    
    fig, ax = plt.subplots()
    title = algo_name + " : Learning Curve - "+ "Dataset-" + str(dataset)
    plt.title(title)
    plt.xlabel("Training Data Percentage (%) --> ")
    plt.ylabel("Score (%) --> ")
    plt.plot(x_data, y_data)
    plt.legend(['train score', 'test score'])
    plt.savefig(figure_name)
    #plt.show()


def Kmeans(dataset=1):
    X, y, dataset_name = GetData(dataset)
    model = KMeans(n_clusters=3, random_state=42)
    data_kmeans = model.fit(X)
    print("\n KMEANS \n")
    print("kmeans labels", data_kmeans.labels_)
    #print("cluster centers", data_kmeans.cluster_centers_)

    for i in range(len(y)):
        #print("labeled data",y[i])
        #print("kmeans data",data_kmeans.labels_[i])
        #print("\n---------------\n")
        if i>=100:
            break

    output = np.asarray([[i,data_kmeans.labels_[i]] for i in range(len(data_kmeans.labels_))])
    labels = np.asarray(data_kmeans.labels_)
    df = pd.DataFrame(output, columns =['Datapoint #', 'K-means-cluster Label'])
    name="k-means labels_data_" + str(dataset) + ".csv"
    df.to_csv(name,index=False)

    fig, ax = plt.subplots()
    title = "K means Cluster"+ "Dataset-" + str(dataset)
    figure_name = "K means Cluster"+ "Dataset-" + str(dataset)
    plt.title(title)
    plt.xlabel("Datapoint # --> ")
    plt.ylabel("K means Cluster label --> ")
    plt.plot(labels)
    #plt.legend()
    plt.savefig(figure_name)
    print("\nsilhouette score", silhouette_score(X,labels))
    print("Calinski Harabaz Index",calinski_harabasz_score(X,labels))
    print("Davies Bouldin score", davies_bouldin_score(X,labels))


    print("preexisting labels")
    print("\nsilhouette score", silhouette_score(X,y))
    print("Calinski Harabaz Index",calinski_harabasz_score(X,y))
    print("Davies Bouldin score", davies_bouldin_score(X,y))

    
    print("\n ------------------------- \n")

def ExpectationMaximization(dataset=1):
    X, y, dataset_name = GetData(dataset)
    model = GaussianMixture(n_components=3, random_state=0)
    results = model.fit(X)
    yhat = model.predict(X)
    print("\n Expectation Maximization \n")
    print("EM weights", results.weights_)
    print("EM means", results.means_)
    print("predictions- is this labels?", yhat)
    #scores=model.score(X)
    #print("scores",score)

    df = pd.DataFrame(yhat, columns =['EM pred Label'])
    name = "EMpreds_ata_" + str(dataset) + ".csv"
    df.to_csv(name,index=False)

    
    print("\nsilhouette score", silhouette_score(X,yhat))
    print("Calinski Harabaz Index",calinski_harabasz_score(X,yhat))
    print("Davies Bouldin score", davies_bouldin_score(X,yhat))
    print("\n ------------------------- \n")
    

    

    """
    Gaussian_nr = 1
    
    for mu, sd, p in zip(results.means_.flatten(), np.sqrt(results.covariances_.flatten()), results.weights_):
        print('Normal_distb {:}: μ = {:.2}, σ = {:.2}, weight = {:.2}'.format(Gaussian_nr, mu, sd, p))
        g_s = stats.norm(mu, sd).pdf(x) * p
        plt.plot(x, g_s, label='gaussian sklearn');
        Gaussian_nr += 1

    sns.distplot(data, bins=20, kde=False, norm_hist=True)
    gmm_sum = np.exp([gmm.score_samples(e.reshape(-1, 1)) for e in x]) 
    plt.plot(x, gmm_sum, label='gaussian mixture');
    plt.legend();
    """


def PCA_analysis(dataset=1,n=2):
    print("PCA with n=",n)
    X, y, dataset_name = GetData(dataset)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    #normalize the dataset
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)

    #run PCA
    if n==1000:
        pca = PCA()
    else:
        pca = PCA(n)
    X_test_initial = X_test
    X_train_initial = X_train
    X_train = pca.fit_transform(X_train)
    X_test = pca.transform(X_test)
    X_test_final = X_test
    X_train_final = X_train

    if n==3:
        print("initial and tansformed")
        for i in range(2):
            print("train initial",X_train_initial[i])
            print("train transormed",X_train_final[i])
            print("test initial",X_test_initial[i])
            print("test transformed",X_test_final[i])
            

    explained_variance = pca.explained_variance_ratio_
    print("vaiances",explained_variance)

    if n==1000:
        df = pd.DataFrame(np.asarray(explained_variance))
        name="PCA_variances_" + str(dataset) + ".csv"
        df.to_csv(name,index=False)
        fig, ax = plt.subplots()
        title = "PCA variances "+ "Dataset-" + str(dataset)
        figure_name = "PCA_variances_"+ "Dataset-" + str(dataset)
        plt.title(title)
        plt.xlabel("PCA # --> ")
        plt.ylabel("Variance --> ")
        plt.plot(explained_variance)        
        plt.savefig(figure_name)

    

    #feats = pca.get_feature_names_out()
    #print("features")

    classifier = RandomForestClassifier(max_depth=2, random_state=0)
    classifier.fit(X_train, y_train)

    # Predicting the Test set results
    y_pred = classifier.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    print('Accuracy',acc)
    print("\n-----")

    plt.plot(explained_variance)

    print("features_out",pca.get_feature_names_out)

    return acc


def PCA_accuracy(dataset=1):
    pca_acc = []
    #PCA_analysis(dataset,1000)
    for i in range(1,11,1):
        pca_acc.append(PCA_analysis(dataset,i))

    #df = pd.DataFrame(np.asarray(pca_acc))
    #name="PCA_accuracy_dataset" + str(dataset) + ".csv"
    #df.to_csv(name,index=False)

    fig, ax = plt.subplots()
    title = "PCA accuracy "+ "Dataset-" + str(dataset)
    figure_name = "PCA_accuracy_"+ "Dataset-" + str(dataset)
    plt.title(title)
    plt.xlabel("PCA # --> ")
    plt.ylabel("Accuracy --> ")
    plt.plot(np.asarray(pca_acc))        
    plt.savefig(figure_name)





def ICA_analysis(dataset=1,n=2):
    print("ICA with n=",n)
    X, y, dataset_name = GetData(dataset)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    #normalize the dataset
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)

    #run ICA
    if n==1000:
        ica = FastICA()
    else:
        ica = FastICA(n)
    X_test_initial = X_test
    X_train_initial = X_train
    X_train = ica.fit_transform(X_train)
    X_test = ica.transform(X_test)
    X_test_final = X_test
    X_train_final = X_train

    if n==3:
        print("initial and tansformed")
        for i in range(2):
            print("train initial",X_train_initial[i])
            print("train transormed",X_train_final[i])
            print("test initial",X_test_initial[i])
            print("test transformed",X_test_final[i])
            

    classifier = RandomForestClassifier(max_depth=2, random_state=0)
    classifier.fit(X_train, y_train)

    # Predicting the Test set results
    y_pred = classifier.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    print('Accuracy',acc)
    print("\n-----")

    return acc

def ICA_accuracy(dataset=1):
    ica_acc = []
    ICA_analysis(dataset,1000)
    for i in range(1,11,1):
        ica_acc.append(ICA_analysis(dataset,i))

    #df = pd.DataFrame(np.asarray(pca_acc))
    #name="PCA_accuracy_dataset" + str(dataset) + ".csv"
    #df.to_csv(name,index=False)

    fig, ax = plt.subplots()
    title = "ICA accuracy "+ "Dataset-" + str(dataset)
    figure_name = "ICA_accuracy_"+ "Dataset-" + str(dataset)
    plt.title(title)
    plt.xlabel("ICA # --> ")
    plt.ylabel("Accuracy --> ")
    plt.plot(np.asarray(ica_acc))        
    plt.savefig(figure_name)


    

def rp_analysis(dataset=1,n=2):
    print("RP with n=",n)
    X, y, dataset_name = GetData(dataset)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    #normalize the dataset
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)

    #run ICA
    if n==1000:
        rp = random_projection.GaussianRandomProjection()
    else:
        rp = random_projection.GaussianRandomProjection(n)
    X_test_initial = X_test
    X_train_initial = X_train
    X_train = rp.fit_transform(X_train)
    X_test = rp.transform(X_test)
    X_test_final = X_test
    X_train_final = X_train

    if n==3:
        print("initial and tansformed")
        for i in range(2):
            print("train initial",X_train_initial[i])
            print("train transormed",X_train_final[i])
            print("test initial",X_test_initial[i])
            print("test transformed",X_test_final[i])
            

    classifier = RandomForestClassifier(max_depth=2, random_state=0)
    classifier.fit(X_train, y_train)

    # Predicting the Test set results
    y_pred = classifier.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    print('Accuracy',acc)
    print("\n-----")

    return acc



def rp_accuracy(dataset=1):
    rp_acc = []
    ##rp_analysis(dataset,1000)
    for i in range(1,20,1):
        rp_acc.append(rp_analysis(dataset,i))


    fig, ax = plt.subplots()
    title = "RP accuracy "+ "Dataset-" + str(dataset)
    figure_name = "RP_accuracy_"+ "Dataset-" + str(dataset)
    plt.title(title)
    plt.xlabel("RP # --> ")
    plt.ylabel("Accuracy --> ")
    plt.plot(np.asarray(rp_acc))        
    plt.savefig(figure_name)


def svd_analysis(dataset=1,n=2):
    print("SVD with n=",n)
    X, y, dataset_name = GetData(dataset)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    #normalize the dataset
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)

    #run ICA
    if n==1000:
        svd = TruncatedSVD()
    else:
        svd = TruncatedSVD(n)
    X_test_initial = X_test
    X_train_initial = X_train
    X_train = svd.fit_transform(X_train)
    X_test = svd.transform(X_test)
    X_test_final = X_test
    X_train_final = X_train

    if n==3:
        print("initial and tansformed")
        for i in range(2):
            print("train initial",X_train_initial[i])
            print("train transormed",X_train_final[i])
            print("test initial",X_test_initial[i])
            print("test transformed",X_test_final[i])
            

    classifier = RandomForestClassifier(max_depth=2, random_state=0)
    classifier.fit(X_train, y_train)

    # Predicting the Test set results
    y_pred = classifier.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    print('Accuracy',acc)
    print("\n-----")

    return acc


def svd_accuracy(dataset=1):
    svd_acc = []
    
    for i in range(1,20,1):
        svd_acc.append(svd_analysis(dataset,i))


    fig, ax = plt.subplots()
    title = "SVD accuracy "+ "Dataset-" + str(dataset)
    figure_name = "SVD_accuracy_"+ "Dataset-" + str(dataset)
    plt.title(title)
    plt.xlabel("SVD # --> ")
    plt.ylabel("Accuracy --> ")
    plt.plot(np.asarray(svd_acc))        
    plt.savefig(figure_name)



def K_means_reduced_algo(dataset=1, algo='pca', n=2):
    
    X, y, dataset_name = GetData(dataset)

    #normalize the dataset
    sc = StandardScaler()
    X = sc.fit_transform(X)
    X = sc.transform(X)

    if algo == 'pca':
        dim_red_model = PCA(n)
        
    elif algo == 'ica':
        dim_red_model = FastICA(n)

    elif algo == 'rp':
        dim_red_model = random_projection.GaussianRandomProjection(n)
        
    elif algo == 'svd':
        dim_red_model = TruncatedSVD(n)
        
        
    X = dim_red_model.fit_transform(X)
    
    model = KMeans(n_clusters=3, random_state=42)
    data_kmeans = model.fit(X)
    print("\n KMEANS for ", algo)


    output = np.asarray([[i,data_kmeans.labels_[i]] for i in range(len(data_kmeans.labels_))])
    labels = np.asarray(data_kmeans.labels_)

    df = pd.DataFrame(output, columns =['Datapoint #', 'K-means-cluster Label'])
    name="k-means reduced labels_data_" + str(dataset) + ".csv"
    df.to_csv(name,index=False)

   
    s=silhouette_score(X,labels)
    c=calinski_harabasz_score(X,labels)
    d=davies_bouldin_score(X,labels)
    
    print("\nsilhouette score", s)
    print("Calinski Harabaz Index",c)
    print("Davies Bouldin score", d)
    print("\n ------------------------- \n")

    return(s,c,d)
    


def K_means_reduced(dataset=1):
    algos = ['pca','ica','rp','svd']
    data=[]
    data.append(np.asarray(["algo", "number of clusters", "sil","cal","dav"]))

    n=3

    for algo in algos:
        (s,c,d)= K_means_reduced_algo(dataset,algo,n)
        data.append(np.asarray([algo,n,s,c,d]))

    df = pd.DataFrame(np.asarray(data))
    name="k-means scores" + str(dataset) + ".csv"
    df.to_csv(name,index=False)




def EM_reduced_algo(dataset=1, algo='pca', n=2):
    
    X, y, dataset_name = GetData(dataset)

    #normalize the dataset
    sc = StandardScaler()
    X = sc.fit_transform(X)
    X = sc.transform(X)

    if algo == 'pca':
        dim_red_model = PCA(n)
        
    elif algo == 'ica':
        dim_red_model = FastICA(n)

    elif algo == 'rp':
        dim_red_model = random_projection.GaussianRandomProjection(n)
        
    elif algo == 'svd':
        dim_red_model = TruncatedSVD(n)
        
        
    X = dim_red_model.fit_transform(X)
    
    model = GaussianMixture(n, random_state=0)
    results = model.fit(X)
    yhat = model.predict(X)
    
    print("\n EM for ", algo)


    #output = np.asarray([[i,data_kmeans.labels_[i]] for i in range(len(data_kmeans.labels_))])
    #labels = np.asarray(data_kmeans.labels_)

   
    s=silhouette_score(X,yhat)
    c=calinski_harabasz_score(X,yhat)
    d=davies_bouldin_score(X,yhat)
    
    print("\nsilhouette score", s)
    print("Calinski Harabaz Index",c)
    print("Davies Bouldin score", d)
    print("\n ------------------------- \n")

    return(s,c,d)
    


def EM_reduced(dataset=1):
    algos = ['pca','ica','rp','svd']
    data=[]
    data.append(np.asarray(["algo", "number of clusters", "sil","cal","dav"]))

    n=3

    for algo in algos:
        (s,c,d)= EM_reduced_algo(dataset,algo,n)
        data.append(np.asarray([algo,n,s,c,d]))

    df = pd.DataFrame(np.asarray(data))
    name="EM scores" + str(dataset) + ".csv"
    df.to_csv(name,index=False)



def NeuralNetOnDimReduced(dataset=1):
    algos = ['pca','ica','rp','svd']

    

    for algo in algos:

        X, y, dataset_name = GetData(dataset)

        n=3
        
        #normalize the dataset
        sc = StandardScaler()
        X = sc.fit_transform(X)
        X = sc.transform(X)

        if algo == 'pca':
            dim_red_model = PCA(n)
            
        elif algo == 'ica':
            dim_red_model = FastICA(n)

        elif algo == 'rp':
            dim_red_model = random_projection.GaussianRandomProjection(n)
            
        elif algo == 'svd':
            dim_red_model = TruncatedSVD(n)
            
            
        X = dim_red_model.fit_transform(X)
        Neural_Net(X,y,algo,dataset)



def NeuralClock(dataset=1):
    algos = ['pca','ica','rp','svd']

    

    for algo in algos:

        X, y, dataset_name = GetData(dataset)

        n=3
        
        #normalize the dataset
        sc = StandardScaler()
        X = sc.fit_transform(X)
        X = sc.transform(X)

        if algo == 'pca':
            dim_red_model = PCA(n)
            
        elif algo == 'ica':
            dim_red_model = FastICA(n)

        elif algo == 'rp':
            dim_red_model = random_projection.GaussianRandomProjection(n)
            
        elif algo == 'svd':
            dim_red_model = TruncatedSVD(n)
            
            
        X = dim_red_model.fit_transform(X)
        print("algo",algo)
        Wallclock(X,y,dataset=1)

        print("no algo")
        X, y, dataset_name = GetData(dataset)
        Wallclock(X,y,dataset=1)






def NeuralNetOnDimReduced_withKmeans(dataset=1):
    algos = ['pca','ica','rp','svd']

    

    for algo in algos:

        X, y, dataset_name = GetData(dataset)

        n=3
        
        #normalize the dataset
        sc = StandardScaler()
        X = sc.fit_transform(X)
        X = sc.transform(X)

        if algo == 'pca':
            dim_red_model = PCA(n)
            
        elif algo == 'ica':
            dim_red_model = FastICA(n)

        elif algo == 'rp':
            dim_red_model = random_projection.GaussianRandomProjection(n)
            
        elif algo == 'svd':
            dim_red_model = TruncatedSVD(n)
            
            
        X = dim_red_model.fit_transform(X)

        #applying kmeans
        model = KMeans(n_clusters=3, random_state=42)
        data_kmeans = model.fit(X)
        print("\n KMEANS for ", algo)


        output = np.asarray([[i,data_kmeans.labels_[i]] for i in range(len(data_kmeans.labels_))])
        labels = np.asarray(data_kmeans.labels_)

        #add labels to X
        print("shapes",X,labels)

        labels_amended = np.ones((len(labels),1))
        labels_amended[:,0] = labels 
        XlabelsAdded = np.hstack((X,labels_amended))
        
        names = algo+"_kmeans_"
        Neural_Net(XlabelsAdded,y,names,dataset)



def NeuralNetOnDimReduced_withEM(dataset=1):
    algos = ['pca','ica','rp','svd']

    

    for algo in algos:

        X, y, dataset_name = GetData(dataset)

        n=3
        
        #normalize the dataset
        sc = StandardScaler()
        X = sc.fit_transform(X)
        X = sc.transform(X)

        if algo == 'pca':
            dim_red_model = PCA(n)
            
        elif algo == 'ica':
            dim_red_model = FastICA(n)

        elif algo == 'rp':
            dim_red_model = random_projection.GaussianRandomProjection(n)
            
        elif algo == 'svd':
            dim_red_model = TruncatedSVD(n)
            
            
        X = dim_red_model.fit_transform(X)

        #applying EM
        model = GaussianMixture(n, random_state=0)
        results = model.fit(X)
        yhat = model.predict(X)
    

        print("\n EM for for ", algo)




        #add labels to X
        print("shapes",X,yhat)

        labels_amended = np.ones((len(yhat),1))
        labels_amended[:,0] = yhat
        XlabelsAdded = np.hstack((X,labels_amended))
        
        names = algo+"_EM_"
        Neural_Net(XlabelsAdded,y,names,dataset)


def FindOptimalK(dataset=1):
    X, y, dataset_name = GetData(dataset)
    wcss = []
    fig, ax = plt.subplots()
    for i in range(1, 11):
        kmeans = KMeans(n_clusters = i, init = 'k-means++', 
                    random_state = 42)
        kmeans.fit(X)
        wcss.append(kmeans.inertia_)
    plt.plot(range(1, 11), wcss)
    title = 'The Elbow Method - Dataset ' + str(dataset)
    plt.title(title)
    plt.xlabel('Number of clusters')
    plt.ylabel('WCSS')
    name = "elbow_dataset" + str(dataset)
    plt.savefig(name)
        
    
        
        
        
    
    


    


def GetData(dataset = 1):
    """
    with open("./fetal_health.csv", 'r') as file:
      csvreader = csv.reader(file)
      for row in csvreader:
        print(row)
    """
    if dataset == 1:
        name = "fetal_health"
    else:
        name = "mobile_price"
    name = name + ".csv"
    df = pd.read_csv(name)
    data = df.to_numpy() # rows and columns just like .csv
    X = data[:,0:-1]
    y = np.transpose(data)[-1]
    #print("data",data)
    #print("X",X)
    #print("y",y)
    return (X,y,name)





def ReadCsv():    
    with open("./fetal_health.csv", 'r') as file:
      csvreader = csv.reader(file)
      for row in csvreader:
        print(row)
    



    

    


if __name__=="__main__":
    print("One day you'll look back on this and smile. \
    There will be tears, but they will be tears of joy")
    #Kmeans()
    #ExpectationMaximization(1)
    #Kmeans(2)
    #ExpectationMaximization(2)

    #PCA_accuracy(1)
    #PCA_accuracy(2)

    #ICA_accuracy(1)
    #ICA_accuracy(2)

    #rp_accuracy(1)
    #rp_accuracy(2)

    #svd_accuracy(1)
    #svd_accuracy(2)

    #K_means_reduced(1)
    #K_means_reduced(2)

    NeuralClock(1)
    NeuralClock(2)

    #EM_reduced(1)
    #EM_reduced(2)
    
    #NeuralNetOnDimReduced(1)
    #NeuralNetOnDimReduced(2)

    #NeuralNetOnDimReduced_withKmeans(1)
    #NeuralNetOnDimReduced_withKmeans(2)

    #NeuralNetOnDimReduced_withEM(1)
    #NeuralNetOnDimReduced_withEM(2)

    #FindOptimalK(1)
    #FindOptimalK(2)
    
    """
    PCA_analysis(1,1000)
    PCA_analysis(1,2)
    PCA_analysis(1,3)
    PCA_analysis(1,4)
    PCA_analysis(1,5)
    PCA_analysis(1,6)
    PCA_analysis(1,7)
    PCA_analysis(1,8)
    PCA_analysis(1,9)
    PCA_analysis(1,10)

    PCA_analysis(2,1000)
    PCA_analysis(2,2)
    PCA_analysis(2,3)
    PCA_analysis(2,4)
    PCA_analysis(2,5)
    PCA_analysis(2,6)
    PCA_analysis(2,7)
    PCA_analysis(2,8)
    PCA_analysis(2,9)
    PCA_analysis(2,10)
    """
    

