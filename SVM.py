import numpy as np
import warnings

warnings.filterwarnings('ignore')


def add_bias_feature(a):
    a_extended = np.zeros((a.shape[0], a.shape[1] + 1))
    a_extended[:, :-1] = a
    a_extended[:, -1] = int(1)
    return a_extended


class ClassifierSVM(object):

    def __init__(self, learning_rate=0.01, constant_decrease_weights=0.1, epochs=200):
        self._learning_rate = learning_rate
        self._constant_decrease_weights = constant_decrease_weights
        self._epochs = epochs
        self._w = None
        self.history_w = []
        self.train_errors = None
        self.val_errors = None
        self.train_loss = None
        self.val_loss = None

    def fit(self, X_train, Y_train, X_val, Y_val, verbose=False):
        if len(set(Y_train)) != 2 or len(set(Y_val)) != 2:
            raise ValueError("Number of classes in Y is not equal 2!")

        X_train = add_bias_feature(X_train)
        X_val = add_bias_feature(X_val)

        self._w = np.random.normal(0.0, 0.05, size=X_train.shape[1])
        self.history_w.append(self._w)
        train_errors = []
        val_errors = []
        train_loss_epochs = []
        val_loss_epochs = []

        for e in range(self._epochs + 1):
            tr_err = 0
            val_err = 0
            tr_loss = 0
            val_loss = 0
            for i, x in enumerate(X_train):
                margin = Y_train[i]*np.dot(self._w, X_train[i])
                if margin >= 1:
                    self._w -= (self._learning_rate * self._constant_decrease_weights * self._w / self._epochs)
                    tr_loss += self.soft_margin(X_train[i], Y_train[i])
                else:
                    self._w += self._learning_rate * (Y_train[i] * X_train[i] - self._constant_decrease_weights * self._w / self._epochs)
                    tr_err += 1
                    tr_loss += self.soft_margin(X_train[i], Y_train[i])
                self.history_w.append(self._w)
            for i, x in enumerate(X_val):
                val_loss += self.soft_margin(X_val[i], Y_val[i])
                val_err += (Y_val[i] * np.dot(self._w, X_val[i]) < 1).astype(int)
            if verbose:
                print(f'Epoch: {e}, Errors: {tr_err}, Margin: {tr_loss}')

            train_errors.append(tr_err)
            val_errors.append(val_err)
            train_loss_epochs.append(tr_loss)
            val_loss_epochs.append(val_loss)

        self.history_w = np.array(self.history_w)
        self.train_errors = np.array(train_errors)
        self.val_errors = np.array(val_errors)
        self.train_loss = np.array(train_loss_epochs)
        self.val_loss = np.array(val_loss_epochs)

    def predict(self, X:np.array):
        y_pred = []
        X_extended = add_bias_feature(X)
        for i in range(len(X_extended)):
            y_pred.append(np.sign(np.dot(self._w, X_extended[i])))
        return np.array(y_pred)

    def loss_function(self, x, y):
        return max(0, 1 - y*np.dot(x, self._w))

    def soft_margin(self, x, y):
        return self.loss_function(x, y) + self._learning_rate * np.dot(self._w, self._w)


















# w*x - b = 1

# w*x - b = 0

# w*x - b = -1














# if __name__ == '__main__':
#     iris = load_iris()
#     X = iris.data
#     Y = iris.target
#     print(X)
#     print(Y)
#     pca = PCA(n_components=2)
#     X = pca.fit_transform(X)
#     Y = (Y > 0).astype(int)*2-1
#     print(X)
#     print(Y)
#
#     X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.4, random_state=2020)
#
#     svm = CustomSVM(etha=0.005, alpha=0.006, epochs=150)
#     svm.fit(X_train, Y_train, X_test, Y_test, verbose=True)
#
#     # print(svm.train_errors)
#     #     # print(svm._w)
#     # print(svm.predict(X_test))
#     print(f"Accuracy: {accuracy_score(Y_test, svm.predict(X_test).flatten()) * 100}%")
#
#     plt.plot(svm.train_loss, linewidth=2, label='train_loss')
#     plt.plot(svm.val_loss, linewidth=2, label='test_loss')
#     plt.grid()
#     plt.legend(prop={'size': 19})
#     plt.show()
#     d = {-1:'green', 1:'red'}
#
#     plt.scatter(X_train[:, 0], X_train[:, 1], c=[d[y] for y in Y_train])
#
#     newline([0, -svm._w[2] / svm._w[1]], [-svm._w[2] / svm._w[0], 0], 'blue')
#     newline([0, 1 / svm._w[1]-svm._w[2] / svm._w[1]], [1 / svm._w[0] - svm._w[2] / svm._w[0], 0], linestyle='dashed') #w0*x_i[0]+w1*x_i[1]+w2*1=1
#     newline([0, -1 / svm._w[1] - svm._w[2] / svm._w[1]], [-1 / svm._w[0] - svm._w[2] / svm._w[0], 0], linestyle='dashed') #w0*x_i[0]+w1*x_i[1]+w2*1=-1
#     plt.grid()
#     plt.show()


    # iris = load_iris()
    # X = iris.data
    # Y = iris.target
    #
    # pca = PCA(n_components=2)
    # X = pca.fit_transform(X)
    # Y = (Y == 2).astype(int) * 2 - 1  # [0,1,2] --> [False,False,True] --> [0,1,1] --> [0,0,2] --> [-1,1,1]
    #
    # X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.4, random_state=2020)
    # svm = CustomSVM(etha=0.03, alpha=0.0001, epochs=300)
    # svm.fit(X_train, Y_train, X_test, Y_test)
    #
    # print(svm.train_errors[:150])  # numbers of error in each epoch
    # print(svm._w)  # w0*x_i[0]+w1*x_i[1]+w2=0
    #
    # plt.plot(svm.train_loss, linewidth=2, label='train_loss')
    # plt.plot(svm.val_loss, linewidth=2, label='test_loss')
    # plt.grid()
    # plt.legend(prop={'size': 15})
    # plt.show()
    #
    # d = {-1: 'green', 1: 'red'}
    # plt.scatter(X_train[:, 0], X_train[:, 1], c=[d[y] for y in Y_train])
    # newline([0, -svm._w[2] / svm._w[1]], [-svm._w[2] / svm._w[0], 0], 'blue')  # в w0*x_i[0]+w1*x_i[1]+w2*1=0 поочередно
    # # подставляем x_i[0]=0, x_i[1]=0
    # newline([0, 1 / svm._w[1] - svm._w[2] / svm._w[1]],
    #         [1 / svm._w[0] - svm._w[2] / svm._w[0], 0])  # w0*x_i[0]+w1*x_i[1]+w2*1=1
    # newline([0, -1 / svm._w[1] - svm._w[2] / svm._w[1]],
    #         [-1 / svm._w[0] - svm._w[2] / svm._w[0], 0])  # w0*x_i[0]+w1*x_i[1]+w2*1=-1
    # plt.show()



    # arr = np.array([[2, 3, 4], [6, 4, 9]])
    #
    # print(arr[:, 0])