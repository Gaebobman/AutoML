# PLEASE WRITE THE GITHUB URL BELOW!
# https://github.com/Gaebobman/AutoML

import sys
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

def load_dataset(dataset_path):
	return pd.read_csv(dataset_path)


def dataset_stat(dataset_df):
	classes = dataset_df.groupby("target").size()
	# Feature 에서 target 제외 하고 return 해야 함.
	return dataset_df.shape[1] - 1, classes[0], classes[1]


def split_dataset(dataset_df, testset_size):
	# X -> target 제외
	# y -> target 만
	x = dataset_df.drop(columns="target", axis=1)
	y = dataset_df["target"]
	# random_state (Seed) 사용 하지 않았음

	return train_test_split(x, y, test_size=testset_size)


def decision_tree_train_test(x_train, x_test, y_train, y_test):
	# To-Do: return acc, prec, recall
	dt_cls = DecisionTreeClassifier()
	dt_cls.fit(x_train, y_train)
	accuracy = accuracy_score(y_test, dt_cls.predict(x_test))
	precision = precision_score(y_test, dt_cls.predict(x_test))
	_recall = recall_score(y_test, dt_cls.predict(x_test))
	# avoid shadowing by 
	return accuracy, precision, _recall


def random_forest_train_test(x_train, x_test, y_train, y_test):
	# To-Do: return acc, prec, recall
	rf_cls = RandomForestClassifier()
	rf_cls.fit(x_train, y_train)
	predict = rf_cls.predict(x_test)
	accuracy = accuracy_score(y_test, predict)
	precision = precision_score(y_test, predict)
	_recall = recall_score(y_test, predict)

	return accuracy, precision, _recall


def svm_train_test(x_train, x_test, y_train, y_test):
	# To-Do: return acc, prec, recall
	svm_pipe = make_pipeline(
		StandardScaler(),
		SVC()
	)

	svm_pipe.fit(x_train, y_train)
	predict = svm_pipe.predict(x_test)
	accuracy = accuracy_score(y_test, predict)
	precision = precision_score(y_test, predict)
	_recall = recall_score(y_test, predict)

	return accuracy, precision, _recall


def print_performances(acc, prec, recall):
	#Do not modify this function!
	print ("Accuracy: ", acc)
	print ("Precision: ", prec)
	print ("Recall: ", recall)


if __name__ == '__main__':
	#Do not modify the main script!
	data_path = sys.argv[1]
	data_df = load_dataset(data_path)

	n_feats, n_class0, n_class1 = dataset_stat(data_df)
	print ("Number of features: ", n_feats)
	print ("Number of class 0 data entries: ", n_class0)
	print ("Number of class 1 data entries: ", n_class1)

	print ("\nSplitting the dataset with the test size of ", float(sys.argv[2]))
	x_train, x_test, y_train, y_test = split_dataset(data_df, float(sys.argv[2]))

	acc, prec, recall = decision_tree_train_test(x_train, x_test, y_train, y_test)
	print ("\nDecision Tree Performances")
	print_performances(acc, prec, recall)

	acc, prec, recall = random_forest_train_test(x_train, x_test, y_train, y_test)
	print ("\nRandom Forest Performances")
	print_performances(acc, prec, recall)

	acc, prec, recall = svm_train_test(x_train, x_test, y_train, y_test)
	print ("\nSVM Performances")
	print_performances(acc, prec, recall)
