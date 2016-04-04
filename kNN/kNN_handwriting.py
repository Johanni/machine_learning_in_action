from numpy import *
import operator
import matplotlib
import matplotlib.pyplot as plt
from numpy import array
from os import listdir

def img2vector(filename):
    returnVect = zeros((1, 1024))
    fr = open(filename)
    for i in range(32):
        lineStr = fr.readline()
        for j in range(32):
            returnVect[0, 32*i+j] = int(lineStr[j])

    return returnVect

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

def handwritingClassTest():
    hwLabels = []
    trainingFileList = listdir('trainingDigits')
    m = len(trainingFileList)
    trainingMat = zeros((m, 1024))
    for i in range(m):
        fileNameStr = trainingFileList[i]
        fileStr = fileNameStr.split('.')[0]
        classNumStr = int(fileStr.split('_')[0]) # ex. 9_45.txt where its the 45th instance of the digit 9
        hwLabels.append(classNumStr)
        trainingMat[i,:] = img2vector('trainingDigits/%s' % fileNameStr)

    testFileList = listdir('testDigits')
    errorCount = 0.0
    mTest = len(testFileList)
    # Didn't normalize features since all values are already between 0 and 1
    for i in range(mTest):
        fileNameStr = testFileList[i]
        fileStr = fileNameStr.split('.')[0]
        classNumStr = int(fileStr.split('_')[0])
        vectorUnderTest = img2vector('testDigits/%s' % fileNameStr)
        classifierResult = classify0(vectorUnderTest, trainingMat, hwLabels, 3)
        print "the classifier came back with: %d, the real answer is %d" % (classifierResult, classNumStr)
        if (classifierResult != classNumStr): errorCount += 1.0
    print "\nthe total number of errors is: %d" % errorCount
    print "\nthe total error rate is: %f" % (errorCount/float(mTest))

# TODO: use kD-trees to reduce the number of calculations

testVector = img2vector('testDigits/0_13.txt')
print testVector[0, 0:31]
print testVector[0, 32:63]

handwritingClassTest()

# Output: total error rate is: 0.011628
