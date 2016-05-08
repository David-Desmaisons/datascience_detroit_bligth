import pandas as pd
import sqlite3
import numpy as np
from sklearn import cross_validation
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, recall_score, precision_score, roc_auc_score
from sklearn.preprocessing import normalize, scale
from abc import ABCMeta, abstractmethod
from sklearn.tree import DecisionTreeClassifier as DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier


def getPrediction(clf, dataset):
    con, data, target = dataset()
    prediction = cross_validation.cross_val_predict(clf, data, target, cv=5)
    con.close()
    return target, prediction


def lazyGetPredictions(clf, datagetter):
    for dataset in datagetter:
        score = getPrediction(clf,dataset)
        yield score


def getPredictions(clf, datagetter):
    return [score for score in lazyGetPredictions(clf, datagetter)]


def plot_confusion_matrix(cm, title='Confusion matrix', cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(2)
    plt.xticks(tick_marks, ['None bligth','Bligth'], rotation=45)
    plt.yticks(tick_marks, ['None bligth','Bligth'])
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


def printPrediction(pred, clf):
    plt.figure()
    target, prev = pred
    conf = confusion_matrix(target, prev)
    print("accuracy {}".format(accuracy_score(target, prev)))
    print("precision {}".format(precision_score(target, prev)))
    print("recall {}".format(recall_score(target, prev)))
    print("f1 {}".format(f1_score(target, prev)))
    print("Roc {}".format(roc_auc_score(target, prev)))
    print(conf)
    plot_confusion_matrix(conf, clf.__class__.__name__)
    print("")   
    
    
def getMetrics(target, prev):
    precision = precision_score(target, prev)
    recall = recall_score(target, prev)
    f1 = f1_score(target, prev)
    roc = roc_auc_score(target, prev)
    return precision, recall, f1, roc
    

def printPredictions(clf, datagetter):
    predictions = getPredictions(clf, datagetter)
    for pred in predictions:
        printPrediction(pred, clf)


def getdata(dataset):
    con = sqlite3.connect("./data/location.s3db")
    alldata = pd.read_sql_query("SELECT BlightViolationNumber,Demolished from {} order by random()".format(dataset), con)
    return con, alldata.as_matrix(['BlightViolationNumber']), alldata.as_matrix(['Demolished']).reshape(len(alldata))


def getalldata(dataset):
    con = sqlite3.connect("./data/location.s3db")
    alldata = pd.read_sql_query("SELECT BlightViolationNumber, Call311Number, AmbientCrime6, AmbientCrime7, AmbientCall6, AmbientCall7, AmbientViolation6, AmbientViolation7, Demolished from {} order by random()".format(dataset), con)
    parameters = alldata.as_matrix(['BlightViolationNumber', 'Call311Number', 'AmbientCrime6', 'AmbientCrime7', 'AmbientCall6', 'AmbientCall7', 'AmbientViolation6', 'AmbientViolation7'])
    normalized = normalize(parameters, norm='l2')
    return con, normalized, alldata.as_matrix(['Demolished']).reshape(len(alldata))


def getafulldata(dataset):
    con = sqlite3.connect("./data/location.s3db")
    alldata = pd.read_sql_query("SELECT BlightViolationNumber, Call311Number, AmbientCrime8, AmbientCrime7 - AmbientCrime8 as crime7, AmbientCrime6 - AmbientCrime7 as crime6, AmbientCall8, AmbientCall6 - AmbientCall7 as call6, AmbientCall7- AmbientCall8 as call7, AmbientViolation8, AmbientViolation7 - AmbientViolation8 as v7, AmbientViolation6 - AmbientViolation7 as v6, Demolished from {} order by random()".format(dataset), con)
    parameters = alldata.as_matrix(['BlightViolationNumber', 'Call311Number', 'AmbientCrime8', 'crime7', 'crime6', 'AmbientCall8', 'call7', 'call6', 'AmbientViolation8', 'v7', 'v6'])
    normalized = normalize(parameters, norm='l2')
    return con, normalized, alldata.as_matrix(['Demolished']).reshape(len(alldata))



class DataExtractorBase(metaclass=ABCMeta):  
    @abstractmethod
    def getparameters(self):
        pass
    
    @abstractmethod
    def getquery(self):
        pass
    
    @abstractmethod
    def normalize(self, parameters):
        pass
    
    def getdata(self):
        con = sqlite3.connect("./data/location.s3db")
        alldata = pd.read_sql_query(self.getquery(), con)
        parameters = alldata.as_matrix(self.getparameters())
        normalized = self.normalize(parameters)
        return con, normalized, alldata.as_matrix(['Demolished']).reshape(len(alldata))
    
    
class DataExtractor(DataExtractorBase):
    def __init__(self, norm='l2'):
        self.norm = norm
    
    def normalize(self, parameters):
        return normalize(parameters, norm=self.norm) if self.norm is not None else parameters

    
class DataExtractorScaler(DataExtractorBase):    
    def normalize(self, parameters):
        return scale(parameters)

    
class ClassicExtractor(DataExtractor):
    def __init__(self, norm='l2'):
        DataExtractor.__init__(self, norm)
        self.norm = norm
     
    def getparameters(self):
        return ['BlightViolationNumber', 'Call311Number', 'AmbientCrime8', 'crime7', 'crime6', 'AmbientCall8', 'call7', 'call6', 'AmbientViolation8', 'v7', 'v6']
    
    def getquery(self):
        return "SELECT BlightViolationNumber, Call311Number, AmbientCrime8, AmbientCrime7 - AmbientCrime8 as crime7, AmbientCrime6 - AmbientCrime7 as crime6, AmbientCall8, AmbientCall6 - AmbientCall7 as call6, AmbientCall7- AmbientCall8 as call7, AmbientViolation8, AmbientViolation7 - AmbientViolation8 as v7, AmbientViolation6 - AmbientViolation7 as v6, Demolished from {} order by random()"
         


def printPredictionsFactory(clf, factory):
    datasets = ["ConsolidatedData", "ConsolidatedData_2"]
    dataset = [lambda : factory(d) for d in datasets]
    return printPredictions(clf, dataset)


def printPredictionsBasic(clf):
    return printPredictionsFactory(clf, getdata)


def checkPredictionFromFactory(clf, datagetter):
    pred = getPrediction(clf, datagetter)
    printPrediction(pred,clf)
    

def checkPrediction(clf):
    checkPredictionFromFactory(clf, lambda : getdata('Newdata'))
    
    
def checkCompletePrediction(clf):
    checkPredictionFromFactory(clf, lambda : getalldata('Newdata'))


def diplayPredictor(clf):
    print("basic")
    checkPrediction(clf)
    print("complete")
    checkCompletePrediction(clf)
    print("full")
    checkPredictionFromFactory(clf, lambda : getafulldata('Newdata'))
    
    
def dodisplay(name, clf, datagetter):
    print(name)
    pred = getPrediction(clf, datagetter)
    printPrediction(pred, clf)  
    
    
def dodisplayfull(clf):
    datagetter = lambda : getafulldata('Newdata')
    dodisplay('', clf, datagetter)
    
    
def BenchMarks(clfs, datagetter=None):
    if datagetter is None:
        datagetter = lambda : getafulldata('Newdata')
    maxprecision =0
    maxrecall=0
    maxf1 =0
    maxroc =0
    bestprecision= None
    bestrecall= None
    bestf1 = None
    bestroc= None
    for clf in clfs:
        target, prediction = getPrediction(clf, datagetter)
        precision, recall, f1, roc = getMetrics(target, prediction)
        if precision> maxprecision:
            maxprecision = precision
            bestprecision = clf
        if recall> maxrecall:
            maxrecall = recall
            bestrecall = clf  
        if f1> maxf1:
            maxf1 = f1
            bestf1 = clf
        if roc> maxroc:
            maxroc = roc
            bestroc = clf
    dodisplay("precision {}".format(bestprecision.__class__.__name__), bestprecision, datagetter)
    dodisplay("recall {}".format(bestrecall.__class__.__name__), bestrecall, datagetter)
    dodisplay("f1 {}".format(bestf1.__class__.__name__), bestf1, datagetter)
    dodisplay("roc {}".format(bestroc.__class__.__name__), bestroc, datagetter)
    

def BenchMarksdatagetter(clf, datagetter1, datagetter2=None):
    if datagetter2 is None:
        datagetter2 = lambda : getafulldata('Newdata')
    dodisplay("First Option {}".format(clf.__class__.__name__), clf, datagetter1)
    dodisplay("Second Option {}".format(clf.__class__.__name__), clf, datagetter2)


def getparameters(dataextractor):
    con, x, y  = dataextractor.getdata()
    clf = DecisionTreeClassifier()
    clf.fit(x,y)
    con.close()
    return sorted(zip(clf.feature_importances_, dataextractor.getparameters()))
#    return [z for z in zip(clf.feature_importances_, dataextractor.getparameters())]


def getparametersBoost(dataextractor):
    con, x, y  = dataextractor.getdata()
    clf = GradientBoostingClassifier()
    clf.fit(x,y)
    con.close()
    return sorted(zip(clf.feature_importances_, dataextractor.getparameters()))


def BenchMarkRandom(extract, extract2 = None):
    lambdaextract = None
    if extract2 is None:
        lambdaextract = lambda : getafulldata('Newdata')
    else:
        lambdaextract= lambda : extract2.getdata()     
    clf = GradientBoostingClassifier()
    BenchMarksdatagetter(clf, lambda : extract.getdata(), lambdaextract)

    
