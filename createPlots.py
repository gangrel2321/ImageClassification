import matplotlib.pyplot as plt
import numpy as np

dataSize = 1638
total_nn_acc = np.load("./output/total_nn_acc.npy")
total_page_acc = np.load("./output/total_page_acc.npy")
time_nn_acc = np.load("./output/temporal_nn_acc.npy")
time_page_acc = np.load("./output/temporal_page_acc.npy")
#Compound accuracy
plt.figure(1)
plt.xlim(0,dataSize)
plt.ylim(0.6,1)
plt.plot(total_nn_acc)
plt.plot(total_page_acc)
plt.title('Compound Accuracy Over Time')
plt.ylabel('Accuracy')
plt.xlabel('Image')
plt.legend(['CNN', 'PageRank'], loc='lower left')
plt.savefig('./output/CompoundAccuracyTest.png', dpi=500)

#temporal accuracy
plt.figure(2)
plt.ylim(0,1)
objects = ('Plant', 'Cactus', 'Cholla', 'Prickly', 'Tree', 'Birch', 'Conifer', 'Pine', 'Oak')
class_nn_acc = []
avg = 0
for i in range(len(time_nn_acc)):
    if i%182 == 181:
        class_nn_acc.append(avg/182)
        avg = 0
    else:
        avg += time_nn_acc[i]

ypos = np.arange(len(class_nn_acc))
print(ypos)
print(class_nn_acc)
plt.bar(ypos, class_nn_acc)
plt.xticks(ypos, objects)
plt.title('Composite Class Accuracy (CNN)')
plt.ylabel('Accuracy')
plt.savefig('./output/CNNClassAccuracyTest.png', dpi=500)

plt.figure(3)
plt.ylim(0,1)
class_page_acc = []
avg = 0
for i in range(len(time_page_acc)):
    if i%182 == 181:
        class_page_acc.append(avg/182)
        avg = 0
    else:
        avg += time_page_acc[i]

ypos = np.arange(len(class_page_acc))
plt.bar(ypos, class_page_acc, color="orange")
plt.xticks(ypos, objects)
plt.title('Composite Class Accuracy (PageRank)')
plt.ylabel('Accuracy')
plt.savefig('./output/PageClassAccuracyTest.png', dpi=500)
plt.show()
