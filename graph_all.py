import matplotlib.pyplot as plt
import matplotlib
from matplotlib.pyplot import figure
import numpy as np
import pandas as pd


names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
xaxis = ['200', '386', '746', '1439', '2779', '5365']

#2 class
figure(num=None, figsize=(10, 7))
for class1 in range(10):
    for class2 in range(class1 + 1, 10):
        results = pd.read_csv("cifar_results/" + str(class1) + "_vs_" + str(class2) + "cnn.csv", index_col=0)
        results2 = pd.read_csv("cifar_results/" + str(class1) + "_vs_" + str(class2) + ".csv", index_col=0)
        rf_accu = results2.iloc[0]
        res_accu = results2.iloc[1]
        cnn32_accu = results.iloc[2]
        cnn32_2_layer = results.iloc[3]
        plt.clf()
        plt.xlabel('Number of Train Samples')
        plt.ylabel('Accuracies')
        plt.title(names[class1] + ' vs ' + names[class2])
        plt.plot(xaxis, res_accu, label="ResNet18")
        plt.plot(xaxis, rf_accu, label="RF")
        plt.plot(xaxis, cnn32_accu, label="32-Filter CNN")
        plt.plot(xaxis, cnn32_2_layer, label="2 Layer CNN")
        plt.legend(loc="upper left")
        plt.savefig('cifar_results/Accu/' + str(class1) + "_vs_" + str(class2))
        
        
        
        
#4 class
figure(num=None, figsize=(10, 7))
experiments = [(0, 1, 2, 3), (0, 1, 2, 4), (0, 1, 2, 5), (0, 1, 2, 6)\
               , (0, 1, 2, 7), (0, 1, 2, 8), (0, 1, 2, 9)]
xaxis = [200, 386, 746, 1439, 2779, 5365, 10359, 20000]

rf_avg = np.array([0.0] * 8)
resnet_avg = np.array([0.0] * 8)
simeplcnn_avg = np.array([0.0] * 8)
cnn2_avg = np.array([0.0] * 8)
cnn5_avg = np.array([0.0] * 8)

for classes in experiments:
    file_title = str(classes[0])
    graph_title = str(classes[0]) + " (" + names[classes[0]] + ") "
    for j in range(1, len(classes)):
        file_title = file_title + "-" + str(classes[j])
        graph_title = graph_title + " vs " + str(classes[j]) + names[classes[j]]
       
    results = pd.read_csv("cifar_results_4+/" + file_title + ".csv", index_col=0)

    rf_accu = results.iloc[0]
    rf_avg += rf_accu
    res_accu = results.iloc[1]
    resnet_avg += res_accu
    simple_cnn = results.iloc[2]
    simeplcnn_avg += simple_cnn
    cnn_2layer = results.iloc[3]
    cnn2_avg += cnn_2layer
    cnn_5layer = results.iloc[4]
    cnn5_avg += cnn_5layer
    
    plt.xlabel('Number of Train Samples')
    plt.ylabel('Accuracies')
    #plt.title(graph_title)
    plt.plot(xaxis, res_accu, color='blue', alpha=.2)
    plt.plot(xaxis, rf_accu, color = 'green', alpha=.2)
    plt.plot(xaxis, simple_cnn, color='gray', alpha=.2)
    plt.plot(xaxis, cnn_2layer, color='brown', alpha=.2)
    plt.plot(xaxis, cnn_5layer, color='orange', alpha=.2)
    
plt.plot(xaxis, res_accu, color='blue', linewidth=2, label="resnet18")
plt.plot(xaxis, rf_accu, color = 'green', linewidth=2, label="RF")
plt.plot(xaxis, simple_cnn, color='gray', linewidth=2, label="1 layer cnn")
plt.plot(xaxis, cnn_2layer, color='brown', linewidth=2, label="2 layer cnn")
plt.plot(xaxis, cnn_5layer, color='orange', linewidth=2, label="5 layer cnn")    
plt.legend(loc="lower right")

plt.xscale('log')
plt.xticks(xaxis, [str(i) for i in xaxis])
plt.title("4 class classification summaries")
plt.savefig('cifar_results_summary/' + file_title)