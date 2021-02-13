import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import warnings
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from SVM import ClassifierSVM


warnings.filterwarnings('ignore')

plt.rcParams['figure.figsize'] = (8, 6)


def newline(p1, p2, color=None, linestyle=None):
    ax = plt.gca()
    xmin, xmax = ax.get_xbound()

    if p2[0] == p1[0]:
        xmin = xmax = p1[0]
        ymin, ymax = ax.get_ybound()
    else:
        ymax = p1[1] + (p2[1] - p1[1]) / (p2[0] - p1[0]) * (xmax - p1[0])
        ymin = p1[1] + (p2[1] - p1[1]) / (p2[0] - p1[0]) * (xmin - p1[0])

    l = mlines.Line2D([xmin, xmax], [ymin, ymax], color=color, linestyle=linestyle)
    ax.add_line(l)
    return l


if __name__ == '__main__':
    iris = load_iris()
    X = iris.data
    Y = iris.target
    # print(X)
    # print(Y)
    print(X.shape, Y.shape)

    pca = PCA(n_components=2)
    X = pca.fit_transform(X)
    Y = (Y > 0).astype(int)*2-1
    print(X)
    print(Y)

    # if you want to do some experiments with delim. hyperplane position

    # X = X[20:, :]
    # Y = Y[20:]
    #
    # print(X.shape, Y.shape)
    # print(Y)

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.4, random_state=2020)

    svm = ClassifierSVM(learning_rate=0.005, constant_decrease_weights=0.006, epochs=150)
    svm.fit(X_train, Y_train, X_test, Y_test, verbose=True)

    # print(svm.train_errors)
    print('--------------------------------')
    print('')
    print(svm._w)
    print('')
    print('--------------------------------')
    print(f"Versicolour: {-1}, Setosa: {1}")
    print('')
    print(svm.predict(X_test))
    print('')
    print('--------------------------------')
    print('')
    print(f"Accuracy: {accuracy_score(Y_test, svm.predict(X_test).flatten()) * 100}%")

    plt.plot(svm.train_loss, linewidth=2, label='train_loss')
    plt.plot(svm.val_loss, linewidth=2, label='test_loss')
    plt.grid()
    plt.legend(prop={'size': 18})
    plt.show()
    d = {-1:'green', 1:'red'}

    plt.scatter(X_train[:, 0], X_train[:, 1], c=[d[y] for y in Y_train])

    newline([0, -svm._w[2] / svm._w[1]], [-svm._w[2] / svm._w[0], 0], 'blue')
    newline([0, 1 / svm._w[1] - svm._w[2] / svm._w[1]], [1 / svm._w[0] - svm._w[2] / svm._w[0], 0], linestyle='dashed')                         #w0*x_i[0]+w1*x_i[1]+w2*1=1
    newline([0, -1 / svm._w[1] - svm._w[2] / svm._w[1]], [-1 / svm._w[0] - svm._w[2] / svm._w[0], 0], linestyle='dashed')                       #w0*x_i[0]+w1*x_i[1]+w2*1=-1
    plt.grid()
    plt.show()

    print(X.shape, Y.shape)

