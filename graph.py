import matplotlib.pyplot as plt
import matplotlib
from matplotlib.pyplot import figure
import numpy as np
import pandas as pd


xaxis = ['20', '50', '126', '317', '796', '2000']
figure(num=None, figsize=(10, 7))
for class1 in range(10):
    for class2 in range(class1 + 1, 10):
        results = pd.read_csv("cifar_results/" + str(class1) + "_vs_" + str(class2) + "cnn.csv", index_col=0)
        rf_accu = results.iloc[0]
        res_accu = results.iloc[1]
        cnn32_accu = results.iloc[2]
        cnn32_2_layer = results.iloc[3]
        plt.clf()
        plt.xlabel('Number of Train Samples')
        plt.ylabel('Accuracies')
        plt.title('Model Accuracies vs Train Samples')
        plt.plot(xaxis, res_accu, label="ResNet18")
        plt.plot(xaxis, rf_accu, label="RF")
        plt.plot(xaxis, cnn32_accu, label="32-Filter CNN")
        plt.plot(xaxis, cnn32_2_layer, label="2 Layer CNN")
        plt.legend(loc="upper left")
        plt.savefig('cifar_results/Accu/' + str(class1) + "_vs_" + str(class2))


xaxis = ['20', '50', '126', '317', '796', '2000']
figure(num=None, figsize=(10, 7))
resnet_rf_avg = np.array([0.0] * 6)
rf_cnn32_avg = np.array([0.0] * 6)
for class1 in range(10):
    for class2 in range(class1 + 1, 10):
        results = pd.read_csv("cifar_results/" + str(class1) + "_vs_" + str(class2) + "cnn.csv", index_col=0)
        rf_accu = results.iloc[0]
        res_accu = results.iloc[1]
        cnn32_2_layer = results.iloc[3]
        
        resnet_rf_avg += np.array(res_accu) - np.array(rf_accu)
        rf_cnn32_avg += np.array(rf_accu) - np.array(cnn32_2_layer)
plt.clf()
plt.xlabel('Number of Train Samples')
plt.ylabel('Difference')
plt.title('Difference in Model Accuracies')
plt.plot(xaxis, resnet_rf_avg, label="Resnet18 - RF")
plt.plot(xaxis, rf_cnn32_avg, label="RF - 2 Layer CNN")
plt.legend(loc="upper left")
plt.savefig('cifar_results/Accu/' + "Summary")


'''
#convrf vs naiverf
kappas_conv_naive = []
xs = []
for class1 in range(10):
    for class2 in range(class1 + 1, 10):
        results = pd.read_csv("cifar_results/" + str(class1) + "_vs_" + str(class2) + "cnn.csv", index_col=0)
        rf_err = .5 / (1 - results.iloc[0])
        convrf_err = .5 / (1 - results.iloc[1])
        kappa = convrf_err - rf_err
        kappas_conv_naive.extend(kappa)
        xs.extend([1, 2, 3, 4, 5, 6])

ind1 = range(0, len(xs), 8)
ind2 = range(1, len(xs), 8)
ind3 = range(2, len(xs), 8)
ind4 = range(3, len(xs), 8)
ind5 = range(4, len(xs), 8)
ind6 = range(5, len(xs), 8)
ind7 = range(6, len(xs), 8)
ind8 = range(7, len(xs), 8)

plt.clf()
for i in range(45):
    plt.plot(xs[i*8:i*8+8], kappas_conv_naive[i*8:i*8+8])
plt.xticks([1,2, 3, 4, 5, 6, 7, 8], ['100', '194', '372', '720', '1390', '2682', '5180', '10000'])

plt.xlabel('Number of Train Samples')
plt.ylabel('Kappa')
plt.title('Convrf kappa - NaiveRF kappa vs # of Train Samples')
plt.savefig('kappas/convrf-naiverf kappa scatter')

plt.clf()
box1 = [(kappas_conv_naive[i]) for i in ind1]
box2 = [(kappas_conv_naive[i]) for i in ind2]
box3 = [(kappas_conv_naive[i]) for i in ind3]
box4 = [(kappas_conv_naive[i]) for i in ind4]
box5 = [(kappas_conv_naive[i]) for i in ind5]
box6 = [(kappas_conv_naive[i]) for i in ind6]
box7 = [(kappas_conv_naive[i]) for i in ind7]
box8 = [(kappas_conv_naive[i]) for i in ind8]
boxs = [box1, box2, box3, box4, box5, box5, box7, box8]

plt.boxplot(boxs)
plt.xlabel('Number of Train Samples')
plt.ylabel('Kappa')
plt.title('Convrf kappa - NaiveRF kappa vs # of Train Samples')
plt.xticks([1,2, 3, 4, 5, 6, 7, 8], ['100', '194', '372', '720', '1390', '2682', '5180', '10000'])

plt.savefig('kappas/convrf-naiverf kappa box')




kappas_conv_cnn = []
xs = []
for class1 in range(10):
    for class2 in range(class1 + 1, 10):
        results = pd.read_csv("cifar_results/" + str(class1) + "_vs_" + str(class2) + ".csv", index_col=0)
        cnn_err = .5 / (1 - results.iloc[3])
        convrf_err = .5 / (1 - results.iloc[1])
        kappa = convrf_err - cnn_err
        kappas_conv_cnn.extend(kappa)
        xs.extend([1, 2, 3, 4, 5, 6, 7, 8])

ind1 = range(0, len(xs), 8)
ind2 = range(1, len(xs), 8)
ind3 = range(2, len(xs), 8)
ind4 = range(3, len(xs), 8)
ind5 = range(4, len(xs), 8)
ind6 = range(5, len(xs), 8)
ind7 = range(6, len(xs), 8)
ind8 = range(7, len(xs), 8)

plt.clf()
for i in range(45):
    plt.plot(xs[i*8:i*8+8], kappas_conv_cnn[i*8:i*8+8])
plt.xticks([1,2, 3, 4, 5, 6, 7, 8], ['100', '194', '372', '720', '1390', '2682', '5180', '10000'])

plt.xlabel('Number of Train Samples')
plt.ylabel('Kappa')
plt.title('Convrf kappa - SimpleCNN kappa vs # of Train Samples')
plt.savefig('kappas/convrf-SimpleCNN kappa scatter')

plt.clf()
box1 = [(kappas_conv_cnn[i]) for i in ind1]
box2 = [(kappas_conv_cnn[i]) for i in ind2]
box3 = [(kappas_conv_cnn[i]) for i in ind3]
box4 = [(kappas_conv_cnn[i]) for i in ind4]
box5 = [(kappas_conv_cnn[i]) for i in ind5]
box6 = [(kappas_conv_cnn[i]) for i in ind6]
box7 = [(kappas_conv_cnn[i]) for i in ind7]
box8 = [(kappas_conv_cnn[i]) for i in ind8]
boxs = [box1, box2, box3, box4, box5, box5, box7, box8]

plt.boxplot(boxs)
plt.xlabel('Number of Train Samples')
plt.ylabel('Kappa')
plt.title('Convrf kappa - SimpleCNN kappa vs # of Train Samples')
plt.xticks([1,2, 3, 4, 5, 6, 7, 8], ['100', '194', '372', '720', '1390', '2682', '5180', '10000'])

plt.savefig('kappas/convrf-SimpleCNN kappa box')


kappas_conv_cnn32 = []
xs = []
for class1 in range(10):
    for class2 in range(class1 + 1, 10):
        results = pd.read_csv("cifar_results/" + str(class1) + "_vs_" + str(class2) + ".csv", index_col=0)
        cnn32_err = .5 / (1 - results.iloc[5])
        convrf_err = .5 / (1 - results.iloc[1])
        kappa = convrf_err - cnn32_err
        kappas_conv_cnn32.extend(kappa)
        xs.extend([1, 2, 3, 4, 5, 6, 7, 8])

ind1 = range(0, len(xs), 8)
ind2 = range(1, len(xs), 8)
ind3 = range(2, len(xs), 8)
ind4 = range(3, len(xs), 8)
ind5 = range(4, len(xs), 8)
ind6 = range(5, len(xs), 8)
ind7 = range(6, len(xs), 8)
ind8 = range(7, len(xs), 8)

plt.clf()
for i in range(45):
    plt.plot(xs[i*8:i*8+8], kappas_conv_cnn32[i*8:i*8+8])
plt.xticks([1,2, 3, 4, 5, 6, 7, 8], ['100', '194', '372', '720', '1390', '2682', '5180', '10000'])

plt.xlabel('Number of Train Samples')
plt.ylabel('Kappa')
plt.title('Convrf kappa - CNN32_2 kappa vs # of Train Samples')
plt.savefig('kappas/convrf-CNN32_2 kappa scatter')
plt.clf()
box1 = [(kappas_conv_cnn32[i]) for i in ind1]
box2 = [(kappas_conv_cnn32[i]) for i in ind2]
box3 = [(kappas_conv_cnn32[i]) for i in ind3]
box4 = [(kappas_conv_cnn32[i]) for i in ind4]
box5 = [(kappas_conv_cnn32[i]) for i in ind5]
box6 = [(kappas_conv_cnn32[i]) for i in ind6]
box7 = [(kappas_conv_cnn32[i]) for i in ind7]
box8 = [(kappas_conv_cnn32[i]) for i in ind8]
boxs = [box1, box2, box3, box4, box5, box5, box7, box8]

plt.boxplot(boxs)
plt.xlabel('Number of Train Samples')
plt.ylabel('Kappa')
plt.title('Convrf kappa - CNN32_2 kappa vs # of Train Samples')
plt.xticks([1,2, 3, 4, 5, 6, 7, 8], ['100', '194', '372', '720', '1390', '2682', '5180', '10000'])

plt.savefig('kappas/convrf-CNN32_2 kappa box')'''