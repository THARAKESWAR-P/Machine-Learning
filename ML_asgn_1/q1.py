from sklearn.decomposition import PCA
import matplotlib.pyplot as plot
from pprint import pprint

def autolabel(rects):
    """
    Attach a text label above each bar, displaying its height.
    """
    for rect in rects:
        height = rect.get_height()
        axis.annotate('%.3f' % height,
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')


if __name__ == 'main':
    print('+++++++++++++++ q1 ++++++++++++++')
    file = open('iris.data', 'r')
    data = [line.split(',') for line in file.readlines()]
    pprint(data[0])

    pca = PCA(n_components = 4)
    dataset = [[row[i] for i in range(len(data)-1)] for row in data]
    pca.fit(dataset)

    variance = pca.explained_variance_ratio_[:]
    labels = ['PC'+str(i+1) for i in range(len(variance))]

    figure, axis = plot.subplots(figsize=(15,7))
    plot1 = axis.bar(labels, variance)

    axis.plot(labels,variance)
    axis.set_title('Proportion of Variance Explained VS Pricipal Component')
    axis.set_xlabel('Pricipal Component')
    axis.set_ylabel('Proportion of Variance Explained')
    autolabel(plot1) 

    plot.show()

    

