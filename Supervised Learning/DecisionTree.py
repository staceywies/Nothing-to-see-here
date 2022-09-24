import numpy as np


class DTLearner(object):
    """
    This is a Decision Tree Learner.
    :param verbose: If “verbose” is True, your code can print out information for debugging.
        If verbose = False your code should not generate ANY output. When we test your code, verbose will be False.
    :type verbose: bool
    """
    def __init__(self, leaf_size=1, verbose=False):
        """
        Constructor method
        """
        self.leafsize=leaf_size
        pass  # move along, these aren't the drones you're looking for

    def add_evidence(self, data_x, data_y):
        """
        Add training data to learner
        :param data_x: A set of feature values used to train the learner
        :type data_x: numpy.ndarray
        :param data_y: The value we are attempting to predict giv
        en the X data
        :type data_y: numpy.ndarray
        """

        # slap on 1s column so linear regression finds a constant term
        ndata = np.append(data_x, data_y, axis=1)
        self.iter=0
        self.tree = self.build_tree(ndata)
        return self.iter

        # build and save the model
    def build_tree(self, data):
        self.iter+=1
        k= np.all(data[:, -1] == data[0, -1])
        if data.shape[0] <= self.leafsize:
            rvalue= np.mean(data[:,-1])
            if rvalue >=0:
                rvalue=1
            else:
                rvalue=-1
            append = np.array([[-1, rvalue, -1, -1]])
            return append
        if k:
            append = np.array([[-1, data[0, -1], -1, -1]])
            return append
        else:
            cormax=0
            for i in range(data.shape[1] - 2):
                x=data[:,i]
                y=data[:,-1]
                cor=np.corrcoef(x,y)
                corsingle=cor[0,1]
                if abs(corsingle)>cormax:
                    cormax=corsingle
                    index=i
            splitval = np.median(data[:,index])
            leftdata=data[data[:,index]<=splitval]
            rightdata=data[data[:,index]>splitval]
            if ((leftdata.shape[0]==data.shape[0]) or (rightdata.shape[0]==data.shape[0])):
                rvalue= np.mean(data[:,-1])
                if rvalue >=0:
                    rvalue=1
                else:
                    rvalue=-1
                return np.array([[-1, rvalue, -1, -1]])
            lefttree = self.build_tree(leftdata)
            righttree = self.build_tree(rightdata)
            root = [index, splitval, 1, lefttree.shape[0] + 1]
            return np.vstack([root, lefttree, righttree])

    def query(self, points):
        """
        Estimate a set of test points given the model we built.
        :param points: A numpy array with each row corresponding to a specific query.
        :type points: numpy.ndarray
        :return: The predicted result of the input data according to the trained model
        :rtype: numpy.ndarray
        """

        ypredic=np.ones(points.shape[0])
        for x in range(points.shape[0]):
            node = int(0)
            while self.tree[node, 0] != -1:
                factor = int(self.tree[node, 0])
                splitval = self.tree[node, 1]
                if points[x, factor] <= splitval:
                    node = node + int(self.tree[node, 2])
                else:
                    node = node + int(self.tree[node, 3])
            ypredic[x]= self.tree[node, 1]
        return ypredic

