from numpy import *
import operator
import matplotlib
import matplotlib.pyplot as plt
from numpy import array


# noinspection PyPep8Naming,PyPep8Naming
def classify0(inX, dataSet, labels, k):
  dataSetSize = dataSet.shape[0]
  # noinspection PyPep8Naming
  diffMat = tile(inX, (dataSetSize, 1)) - dataSet
  sqDiffMat = diffMat**2
  sqDistances = sqDiffMat.sum(axis=1)
  distances = sqDistances**0.5
  sortedDistIndices = distances.argsort()
  classCount = {}
  for i in range(k):
    voteILabel = labels[sortedDistIndices[i]]
    classCount[voteILabel] = classCount.get(voteILabel, 0) + 1
  sortedClassCount = sorted(classCount.iteritems(),
                            key=operator.itemgetter(1), reverse=True)
  return sortedClassCount[0][0]

def createDataSet():
  group = array([[1.0,1.1],[1.0,1.0],[0,0],[0,0.1]])
  labels = [ 'A', 'A', 'B', 'B']
  return group, labels

def file2matrix(filename):
    fr = open(filename)
    numberOfLines = len(fr.readlines())         #get the number of lines in the file
    returnMat = zeros((numberOfLines,3))        #prepare matrix to return
    classLabelVector = []                       #prepare labels return
    fr = open(filename)
    index = 0
    for line in fr.readlines():
        line = line.strip()
        listFromLine = line.split('\t')
        returnMat[index,:] = listFromLine[0:3]
        classLabelVector.append((listFromLine[-1]))
        index += 1
    return returnMat,classLabelVector

def autoNorm(dataSet):
    minVals = dataSet.min(0)
    maxVals = dataSet.max(0)
    ranges = maxVals - minVals
    normDataSet = zeros(shape(dataSet))
    m = dataSet.shape[0]
    normDataSet = dataSet - tile(minVals, (m,1))
    normDataSet = normDataSet/tile(ranges, (m,1))   #element wise divide
    return normDataSet, ranges, minVals

def datingClassTest():
    hoRatio = 0.10
    datingDataMat, datingLabels = file2matrix('datingTestSet2.txt')
    normMat, ranges, minVals = autoNorm(datingDataMat)
    m = normMat.shape[0]
    numTestVecs = int(m*hoRatio)
    errorCount = 0.0
    for i in range(numTestVecs):
        classifierResult = classify0(normMat[i,:], normMat[numTestVecs:m,:], datingLabels[numTestVecs:m], 3)
        print datingLabels[i]
        print "the classifier came back with: %d, the real answer is %d" % (int(classifierResult), int(datingLabels[i]))
        if (classifierResult != datingLabels[i]): errorCount += 1.0
    print "the total error rate is: %f" % (errorCount/float(numTestVecs))

def classify_person():
    result_list = ['not at all', 'in small doses', 'in large doses']
    percent_tats = float(raw_input('percentage of time spent playing video games?'))
    ff_miles = float(raw_input('frequent flier miles earned per year?'))
    ice_cream = float(raw_input('liters of ice cream consumed per year?'))
    datingDataMat, datingLabels = file2matrix('datingTestSet2.txt')
    normMat, ranges, minVals = autoNorm(datingDataMat)
    inArr = array([ff_miles, percent_tats, ice_cream])
    classifier_result = classify0((inArr - minVals)/ranges, normMat, datingLabels, 3)
    print "You will probably like this person: ", result_list[int(classifier_result) - 1]






group,labels = createDataSet()
print group.shape[1]
classify0([0,0], group, labels, 3)
datingDataMat,datingLabels = file2matrix('datingTestSet2.txt')
print datingDataMat
print datingLabels
fig = plt.figure()
ax = fig.add_subplot(111)
ax.scatter(datingDataMat[:,1], datingDataMat[:,2],
           15.0*array(datingLabels).astype(float), 15.0*array(datingLabels).astype(float))
#plt.show()

normMat, ranges, minVals = autoNorm(datingDataMat)
print normMat
print ranges
print minVals

datingClassTest()
classify_person()