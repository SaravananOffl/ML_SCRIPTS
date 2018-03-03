from sklearn import tree

# height ,weight , shoe size
X = [[181,80,44],[181,60,45],[154,54,23],[145,63,42],
     [170,61,56]]

#corresponding Y values
Y = ['male','female','female','male','male']

#using tree model in scikit learn  http://scikit-learn.org/stable/tutorial/machine_learning_map/
clf = tree.DecisionTreeClassifier()

#fit the X and Y
clf = clf.fit(X,Y)

#we now ask the model to predict acc to input
prediction = clf.predict([[190,61,56]])

print(prediction)