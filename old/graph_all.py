import matplotlib.pyplot as plt
import matplotlib
from matplotlib.pyplot import figure
from itertools import combinations
import numpy as np
import pandas as pd


names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
xaxis = [10, 27, 72, 193, 518, 1389, 3728, 10000] 
    
#2 class
figure(num=None, figsize=(10, 7))
nums = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
experiments = list(combinations(nums, 2))[:45]

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
       
    results = pd.read_csv("cifar_results_fixed/" + file_title + ".csv", index_col=0)

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
    plt.plot(xaxis, res_accu, color='#ff7f00', alpha=.1)
    plt.plot(xaxis, rf_accu, color = '#e41a1c', alpha=.1)
    plt.plot(xaxis, simple_cnn, color='#377eb8', alpha=.1)
    plt.plot(xaxis, cnn_2layer, color='#4daf4a', alpha=.1)
    plt.plot(xaxis, cnn_5layer, color='#984ea3', alpha=.1)
    
plt.plot(xaxis, resnet_avg / len(experiments), color='#ff7f00', linewidth=4, label="resnet18")
plt.plot(xaxis, rf_avg / len(experiments), color = '#e41a1c', linewidth=4, label="RF")
plt.plot(xaxis, simeplcnn_avg / len(experiments), color='#377eb8', linewidth=4, label="1 layer cnn")
plt.plot(xaxis, cnn2_avg / len(experiments), color='#4daf4a', linewidth=4, label="2 layer cnn")
plt.plot(xaxis, cnn5_avg / len(experiments), color='#984ea3', linewidth=4, label="5 layer cnn")    
plt.legend(loc="lower right")
plt.ylim((0, 1))
plt.xscale('log')
plt.xticks(xaxis, [str(i) for i in xaxis])
plt.title("2 class classification summaries")
plt.savefig('cifar_results_fixed/' + "2-class-summary")

        
        
        
#4 class
figure(num=None, figsize=(10, 7))
nums = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
experiments = list(combinations(nums, 4))[:45]

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
       
    results = pd.read_csv("cifar_results_fixed/" + file_title + ".csv", index_col=0)

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
    plt.plot(xaxis, res_accu, color='#ff7f00', alpha=.1)
    plt.plot(xaxis, rf_accu, color = '#e41a1c', alpha=.1)
    plt.plot(xaxis, simple_cnn, color='#377eb8', alpha=.1)
    plt.plot(xaxis, cnn_2layer, color='#4daf4a', alpha=.1)
    plt.plot(xaxis, cnn_5layer, color='#984ea3', alpha=.1)
    
plt.plot(xaxis, resnet_avg / len(experiments), color='#ff7f00', linewidth=4, label="resnet18")
plt.plot(xaxis, rf_avg / len(experiments), color = '#e41a1c', linewidth=4, label="RF")
plt.plot(xaxis, simeplcnn_avg / len(experiments), color='#377eb8', linewidth=4, label="1 layer cnn")
plt.plot(xaxis, cnn2_avg / len(experiments), color='#4daf4a', linewidth=4, label="2 layer cnn")
plt.plot(xaxis, cnn5_avg / len(experiments), color='#984ea3', linewidth=4, label="5 layer cnn")     
plt.legend(loc="lower right")
plt.ylim((0, 1))
plt.xscale('log')
plt.xticks(xaxis, [str(i) for i in xaxis])
plt.title("4 class classification summaries")
plt.savefig('cifar_results_fixed/' + "4-class-summary")



#5 class
figure(num=None, figsize=(10, 7))
nums = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
experiments = list(combinations(nums, 5))[:45]

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
       
    results = pd.read_csv("cifar_results_fixed/" + file_title + ".csv", index_col=0)

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
    plt.plot(xaxis, res_accu, color='#ff7f00', alpha=.1)
    plt.plot(xaxis, rf_accu, color = '#e41a1c', alpha=.1)
    plt.plot(xaxis, simple_cnn, color='#377eb8', alpha=.1)
    plt.plot(xaxis, cnn_2layer, color='#4daf4a', alpha=.1)
    plt.plot(xaxis, cnn_5layer, color='#984ea3', alpha=.1)
    
plt.plot(xaxis, resnet_avg / len(experiments), color='#ff7f00', linewidth=4, label="resnet18")
plt.plot(xaxis, rf_avg / len(experiments), color = '#e41a1c', linewidth=4, label="RF")
plt.plot(xaxis, simeplcnn_avg / len(experiments), color='#377eb8', linewidth=4, label="1 layer cnn")
plt.plot(xaxis, cnn2_avg / len(experiments), color='#4daf4a', linewidth=4, label="2 layer cnn")
plt.plot(xaxis, cnn5_avg / len(experiments), color='#984ea3', linewidth=4, label="5 layer cnn")    
plt.legend(loc="lower right")

plt.xscale('log')
plt.xticks(xaxis, [str(i) for i in xaxis])
plt.title("5 class classification summaries")
plt.savefig('cifar_results_fixed/' + "5-class-summary")


#6 class
figure(num=None, figsize=(10, 7))
nums = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
experiments = list(combinations(nums, 6))[:45]

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
       
    results = pd.read_csv("cifar_results_fixed/" + file_title + ".csv", index_col=0)

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
    plt.plot(xaxis, res_accu, color='#ff7f00', alpha=.1)
    plt.plot(xaxis, rf_accu, color = '#e41a1c', alpha=.1)
    plt.plot(xaxis, simple_cnn, color='#377eb8', alpha=.1)
    plt.plot(xaxis, cnn_2layer, color='#4daf4a', alpha=.1)
    plt.plot(xaxis, cnn_5layer, color='#984ea3', alpha=.1)
    
plt.plot(xaxis, resnet_avg / len(experiments), color='#ff7f00', linewidth=4, label="resnet18")
plt.plot(xaxis, rf_avg / len(experiments), color = '#e41a1c', linewidth=4, label="RF")
plt.plot(xaxis, simeplcnn_avg / len(experiments), color='#377eb8', linewidth=4, label="1 layer cnn")
plt.plot(xaxis, cnn2_avg / len(experiments), color='#4daf4a', linewidth=4, label="2 layer cnn")
plt.plot(xaxis, cnn5_avg / len(experiments), color='#984ea3', linewidth=4, label="5 layer cnn")    
plt.legend(loc="lower right")
plt.ylim((0, 1))
plt.xscale('log')
plt.xticks(xaxis, [str(i) for i in xaxis])
plt.title("6 class classification summaries")
plt.savefig('cifar_results_fixed/' + "6-class-summary")


#7 class
figure(num=None, figsize=(10, 7))
nums = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
experiments = list(combinations(nums, 7))[:45]

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
       
    results = pd.read_csv("cifar_results_fixed/" + file_title + ".csv", index_col=0)

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
    plt.plot(xaxis, res_accu, color='#ff7f00', alpha=.1)
    plt.plot(xaxis, rf_accu, color = '#e41a1c', alpha=.1)
    plt.plot(xaxis, simple_cnn, color='#377eb8', alpha=.1)
    plt.plot(xaxis, cnn_2layer, color='#4daf4a', alpha=.1)
    plt.plot(xaxis, cnn_5layer, color='#984ea3', alpha=.1)
    
plt.plot(xaxis, resnet_avg / len(experiments), color='#ff7f00', linewidth=4, label="resnet18")
plt.plot(xaxis, rf_avg / len(experiments), color = '#e41a1c', linewidth=4, label="RF")
plt.plot(xaxis, simeplcnn_avg / len(experiments), color='#377eb8', linewidth=4, label="1 layer cnn")
plt.plot(xaxis, cnn2_avg / len(experiments), color='#4daf4a', linewidth=4, label="2 layer cnn")
plt.plot(xaxis, cnn5_avg / len(experiments), color='#984ea3', linewidth=4, label="5 layer cnn")     
plt.legend(loc="lower right")
plt.ylim((0, 1))
plt.xscale('log')
plt.xticks(xaxis, [str(i) for i in xaxis])
plt.title("7 class classification summaries")
plt.savefig('cifar_results_fixed/' + "7-class-summary")