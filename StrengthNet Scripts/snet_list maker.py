import numpy as np
import pickle
import itertools
import numpy as np

from sklearn import svm, linear_model
from sklearn.model_selection import KFold


def transform_pairwise(X, y):
    """Transforms data into pairs with balanced labels for ranking
    Transforms a n-class ranking problem into a two-class classification
    problem. Subclasses implementing particular strategies for choosing
    pairs should override this method.
    In this method, all pairs are choosen, except for those that have the
    same target value. The output is an array of balanced classes, i.e.
    there are the same number of -1 as +1
    Parameters
    ----------
    X : array, shape (n_samples, n_features)
        The data
    y : array, shape (n_samples,) or (n_samples, 2)
        Target labels. If it's a 2D array, the second column represents
        the grouping of samples, i.e., samples with different groups will
        not be considered.
    Returns
    -------
    X_trans : array, shape (k, n_feaures)
        Data as pairs
    y_trans : array, shape (k,)
        Output class labels, where classes have values {-1, +1}
    """
    X_new = []
    y_new = []
    y = np.asarray(y)
    if y.ndim == 1:
        y = np.c_[y, np.ones(y.shape[0])]
    comb = itertools.combinations(range(X.shape[0]), 2)
    for k, (i, j) in enumerate(comb):
        if y[i, 0] == y[j, 0] or y[i, 1] != y[j, 1]:
            # skip if same target or different group
            continue
        X_new.append(X[i] - X[j])
        y_new.append(np.sign(y[i, 0] - y[j, 0]))
        # output balanced classes
        if y_new[-1] != (-1) ** k:
            y_new[-1] = - y_new[-1]
            X_new[-1] = - X_new[-1]
    return np.asarray(X_new), np.asarray(y_new).ravel()


class RankSVM(svm.LinearSVC):
    """Performs pairwise ranking with an underlying LinearSVC model
    Input should be a n-class ranking problem, this object will convert it
    into a two-class classification problem, a setting known as
    `pairwise ranking`.
    See object :ref:`svm.LinearSVC` for a full description of parameters.
    """

    def fit(self, X, y):
        """
        Fit a pairwise ranking model.
        Parameters
        ----------
        X : array, shape (n_samples, n_features)
        y : array, shape (n_samples,) or (n_samples, 2)
        Returns
        -------
        self
        """
        X_trans, y_trans = transform_pairwise(X, y)
        super(RankSVM, self).fit(X_trans, y_trans)
        return self

    def predict(self, X):
        """
        Predict an ordering on X. For a list of n samples, this method
        returns a list from 0 to n-1 with the relative order of the rows of X.
        Parameters
        ----------
        X : array, shape (n_samples, n_features)
        Returns
        -------
        ord : array, shape (n_samples,)
            Returns a list of integers representing the relative order of
            the rows in X.
        """
        if hasattr(self, 'coef_'):
            np.argsort(np.dot(X, self.coef_.T))
        else:
            raise ValueError("Must call fit() prior to predict()")

    def score(self, X, y):
        """
        Because we transformed into a pairwise problem, chance level is at 0.5
        """
        X_trans, y_trans = transform_pairwise(X, y)
        return np.mean(super(RankSVM, self).predict(X_trans) == y_trans)



number = '10'
for emotion in ["Happy", "Sad", "Angry", "Surprised"]:
    #test: four[:450]
    #val: four[450:1000]
    #train: five
    data = np.load(
        f'C:/Users/harol/Desktop/Thesis/Comparing Datasets/Data/five_{emotion}.npy', allow_pickle=True)[:20, :]
    print(data.shape)
    # Label 2 is neutral speech, 1 is emotional
    filenames = data[:, -1]
    labels = data[:, -2]

    boolArr = (labels == '1')
    print("Files Left", filenames[boolArr].shape)
    print("bool", filenames[boolArr][0])

    '''for filename in filenames[boolArr]:
        if int(filename[7:11]) > 700:
            print("Problem", filename[7:11])
        if int(filename[7:11]) < 351:
            print("Problem", filename[7:11])'''
    X = np.matrix(data[:, :384])[boolArr].astype("float64")
    print("X Shape", X.shape)

    tot = 0

    w = pickle.load(open(
        f"C:/Users/harol/Desktop/Thesis/SVM Results (trained on one and five)/{emotion}_{number}.sav", 'rb'))
    # Gives a list of a weight for each OpenSmile feature

    score_list = []

    for i in range(X.shape[0]):
        tot += 1
        score_list.append(w.decision_function(X[i].reshape(1, -1)))

    print(len(score_list))
    # print(score_list)

    def NormalizeData(data):
        return (data - np.min(data)) / (np.max(data) - np.min(data))

    result = NormalizeData(np.array(score_list))

    print(result.shape)

    with open(f'C:/Users/harol/Desktop/Thesis/Comparing Datasets/snet_{number}_new/{emotion}_{number}.txt', 'w') as f:
        for i in range(result.shape[0]):
            my_str = ""
            resultstring = str(result[i])
            resultstring = resultstring.strip('[]')

            my_str += str(filenames[boolArr][i][0:4]) + f"/{emotion}/" + \
                str(filenames[boolArr][i]) + ',' + \
                resultstring + "\n"
            f.write(my_str)

    with open(f'C:/Users/harol/Desktop/Thesis/Comparing Datasets/snet_{number}_new/snet_{number}.csv', 'a') as f:
        for i in range(result.shape[0]):
            my_str = ""
            resultstring = str(result[i])
            resultstring = resultstring.strip('[]')

            my_str += str(filenames[boolArr][i][0:4]) + f"/{emotion}/" + \
                str(filenames[boolArr][i]) + ',' + \
                resultstring + "\n"
            f.write(my_str)

'''for emotion in ["Happy", "Sad", "Angry", "Surprised"]:
    #test: four[:450]
    #val: four[450:1000]
    #train: five
    data = np.load(
        f'C:/Users/harol/Desktop/Thesis/Comparing Datasets/Data/four_{emotion}.npy', allow_pickle=True)[:int(number), :]
    print(data.shape)
    # Label 2 is neutral speech, 1 is emotional
    filenames = data[:450, -1]
    labels = data[:450, -2]

    boolArr = (labels == '1')
    print("Files Left", filenames[boolArr].shape)
    print("bool", filenames[boolArr][0])

    for filename in filenames[boolArr]:
        if int(filename[7:11]) > 700:
            print("Problem", filename[7:11])
        if int(filename[7:11]) < 351:
            print("Problem", filename[7:11])
    X = np.matrix(data[:450, :384])[boolArr].astype("float64")
    print("X Shape", X.shape)

    tot = 0
    w = pickle.load(open(
        f"C:/Users/harol/Desktop/Thesis/SVM Results (trained on one and five)/{emotion}_{number}.sav", 'rb'))
    # Gives a list of a weight for each OpenSmile feature
    score_list = []

    for i in range(X.shape[0]):
        tot += 1
        score_list.append(w.decision_function(X[i].reshape(1, -1)))
    print(len(score_list))
    # print(score_list)

    def NormalizeData(data):
        return (data - np.min(data)) / (np.max(data) - np.min(data))

    result = NormalizeData(np.array(score_list))

    print(result.shape)

    with open(f'C:/Users/harol/Desktop/Thesis/Comparing Datasets/snet_{number}_new/{emotion}_{number}_test.txt', 'w') as f:
        for i in range(result.shape[0]):
            my_str = ""
            resultstring = str(result[i])
            resultstring = resultstring.strip('[]')
            print(resultstring)
            my_str += str(filenames[boolArr][i][0:4]) + f"/{emotion}/" + \
                str(filenames[boolArr][i]) + ',' + \
                resultstring + "\n"
            f.write(my_str)

    with open(f'C:/Users/harol/Desktop/Thesis/Comparing Datasets/snet_{number}_new/snet_{number}_test.csv', 'a') as f:
        for i in range(result.shape[0]):
            my_str = ""
            resultstring = str(result[i])
            resultstring = resultstring.strip('[]')
            print(resultstring)
            my_str += str(filenames[boolArr][i][0:4]) + f"/{emotion}/" + \
                str(filenames[boolArr][i]) + ',' + \
                resultstring + "\n"
            f.write(my_str)

for emotion in ["Happy", "Sad", "Angry", "Surprised"]:
    #test: four[:450]
    #val: four[450:1000]
    #train: five
    data = np.load(
        f'C:/Users/harol/Desktop/Thesis/Comparing Datasets/Data/four_{emotion}.npy', allow_pickle=True)[:int(number), :]
    print(data.shape)
    # Label 2 is neutral speech, 1 is emotional
    filenames = data[450:1000, -1]
    labels = data[450:1000, -2]

    boolArr = (labels == '1')
    print("Files Left", filenames[boolArr].shape)
    print("bool", filenames[boolArr][0])

    for filename in filenames[boolArr]:
        if int(filename[7:11]) > 700:
            print("Problem", filename[7:11])
        if int(filename[7:11]) < 351:
            print("Problem", filename[7:11])
    X = np.matrix(data[450:1000, :384])[boolArr].astype("float64")
    print("X Shape", X.shape)

    tot = 0
    w = pickle.load(open(
        f"C:/Users/harol/Desktop/Thesis/SVM Results (trained on one and five)/{emotion}_{number}.sav", 'rb'))
    # Gives a list of a weight for each OpenSmile feature

    score_list = []

    for i in range(X.shape[0]):
        tot += 1
        score_list.append(w.decision_function(X[i].reshape(1, -1)))
    print(len(score_list))
    # print(score_list)

    def NormalizeData(data):
        return (data - np.min(data)) / (np.max(data) - np.min(data))

    result = NormalizeData(np.array(score_list))

    print(result.shape)

    with open(f'C:/Users/harol/Desktop/Thesis/Comparing Datasets/snet_{number}_new/{emotion}_{number}_val.txt', 'w') as f:
        for i in range(result.shape[0]):
            my_str = ""
            resultstring = str(result[i])
            resultstring = resultstring.strip('[]')

            my_str += str(filenames[boolArr][i][0:4]) + f"/{emotion}/" + \
                str(filenames[boolArr][i]) + ',' + \
                resultstring + "\n"
            f.write(my_str)

    with open(f'C:/Users/harol/Desktop/Thesis/Comparing Datasets/snet_{number}_new/snet_{number}_val.csv', 'a') as f:
        for i in range(result.shape[0]):
            my_str = ""
            resultstring = str(result[i])
            resultstring = resultstring.strip('[]')

            my_str += str(filenames[boolArr][i][0:4]) + f"/{emotion}/" + \
                str(filenames[boolArr][i]) + ',' + \
                resultstring + "\n"
            f.write(my_str)'''
