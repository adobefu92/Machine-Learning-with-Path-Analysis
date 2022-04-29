import time 
from sklearn import metrics

def ml_modeling(model_name,X_train, X_test, y_train, y_test):

    start = time.time()

    model=model_name

    if model=="RF":
        from sklearn.ensemble import RandomForestClassifier
        clf=RandomForestClassifier(n_estimators=100)
        clf.fit(X_train,y_train.ravel())
        y_pred=clf.predict(X_test)
    elif model=="CNB":
        from sklearn.naive_bayes import CategoricalNB
        clf = CategoricalNB()
        clf.fit(X_train, y_train.ravel())
        y_pred=clf.predict(X_test)
    elif model=="SVM":    
        from sklearn import svm
        clf = svm.SVC(kernel='linear',probability=True)
        clf.fit(X_train, y_train.ravel())
        y_pred = clf.predict(X_test)
    elif model=="Boost":  
        from sklearn.ensemble import AdaBoostClassifier
        clf = AdaBoostClassifier(n_estimators=100, random_state=0)
        clf.fit(X_train, y_train.ravel())
        y_pred=clf.predict(X_test)
    elif model=="NN":  
        from sklearn.neural_network import MLPClassifier
        clf = MLPClassifier(solver='lbfgs', alpha=1e-5,
                         hidden_layer_sizes=(5, 4), max_iter=2000 ,random_state=1)
        clf.fit(X_train, y_train.ravel())
        y_pred=clf.predict(X_test)
    elif model=="LR":  
        from sklearn.linear_model import LogisticRegression
        clf = LogisticRegression()
        y_pred=clf.predict(X_test)   

    end = time.time()
    print("X+Y1=Y2 Time =")
    print(end-start)

    # Model Accuracy, how often is the classifier correct?
    print(model+" Accuracy:",metrics.accuracy_score(y_test, y_pred))
    
    return(clf)
