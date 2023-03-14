import numpy as np
import matplotlib.pyplot as plt
import math

def generateData():

    xA = np.random.randint(1, 3000, size=(1000, 2))
    xB = np.random.randint(2000, 6000, size=(1000, 2))
    xC = np.random.randint(4000, 7500, size=(1000, 2))

    xMain = np.concatenate((xA, xB, xC), axis=0)
    dataLabel = np.empty(len(xMain), dtype = int)
    dataLabel.fill(9)
    lower = np.min(xMain)
    upper = np.max(xMain)
    print(dataLabel)

    c1 = np.random.randint(lower, upper, size=(1, 2))
    c2 = np.random.randint(lower, upper, size=(1, 2))
    c3 = np.random.randint(lower, upper, size=(1, 2))
    cMain = np.concatenate((c1, c2, c3), axis=0)

    return xMain, cMain, xA, xB, xC, c1, c2, c3, dataLabel


def plotData(xA, xB, xC, c1, c2, c3, xMain, dataLabel):

    plt.plot(xMain[dataLabel==9, 0], xMain[dataLabel==9, 1], 'ko', alpha=0.25)
    plt.plot(xMain[dataLabel==1, 0], xMain[dataLabel==1, 1], 'bo', alpha=0.25)
    plt.plot(xMain[dataLabel==2, 0], xMain[dataLabel==2, 1], 'ro', alpha=0.25)
    plt.plot(xMain[dataLabel==3, 0], xMain[dataLabel==3, 1], 'go', alpha=0.25)
    plt.plot(c1[:, 0], c1[:, 1], 'bo', markersize=8)
    plt.plot(c2[:, 0], c2[:, 1], 'ro', markersize=8)
    plt.plot(c3[:, 0], c3[:, 1], 'go', markersize=8)

    plt.axis('square')
    plt.figure()
    plt.show()


def centroidDistance(x, c):

    dataDistance = np.array(math.sqrt((x[0, 0] - c[0, 0]) ** 2 + (x[0, 1] - c[0, 1]) ** 2))





    for i in range(1, len(x)):
        dataDistance = np.append(dataDistance, (math.sqrt((x[i, 0] - c[0, 0]) ** 2 + (x[i, 1] - c[0, 1]) ** 2)))

    return dataDistance


def labelData(dist1, dist2, dist3):

    dataLabel = np.array(min(dist1[0], dist2[0], dist3[0]))

    if dataLabel == dist1[0]:
        dataLabel = 1
    elif dataLabel == dist2[0]:
        dataLabel = 2
    elif dataLabel == dist3[0]:
        dataLabel = 3

    for i in range(1, len(dist1)):

        dataLabel = np.append(dataLabel, min(dist1[i], dist2[i], dist3[i]))

        if dataLabel[i] == dist1[i]:
            dataLabel[i] = 1
        elif dataLabel[i] == dist2[i]:
            dataLabel[i] = 2
        elif dataLabel[i] == dist3[i]:
            dataLabel[i] = 3

    return dataLabel


def centroidMean(xMain, dataLabel,c1,c2,c3):

    totalC1 = 0
    totalC2 = 0
    totalC3 = 0
    c1x = 0
    c2x = 0
    c3x = 0
    c1y = 0
    c2y = 0
    c3y = 0

    for i in range((len(xMain))):

        if dataLabel[i] == 1:
            totalC1 += 1
            c1x += xMain[i, 0]
            c1y += xMain[i, 1]
        elif dataLabel[i] == 2:
            totalC2 += 1
            c2x += xMain[i, 0]
            c2y += xMain[i, 1]
        elif dataLabel[i] == 3:
            totalC3 += 1
            c3x += xMain[i, 0]
            c3y += xMain[i, 1]

    if totalC1 > 0:
        newC1 = np.array([[c1x / totalC1, c1y / totalC1]])
    else:
        newC1 = c1
    if totalC2 > 0:
        newC2 = np.array([[c2x / totalC2, c2y / totalC2]])
    else:
        newC2 = c2
    if totalC3 >0:
        newC3 = np.array([[c3x / totalC3, c3y / totalC3]])
    else:
        newC3 = c3

    return newC1, newC2, newC3



#client code

xMain, cMain, xA, xB, xC, c1, c2, c3, dataLabel = generateData()
plotData(xA, xB, xC, c1, c2, c3, xMain, dataLabel)
centroidDifferenceC1 = 100
centroidDifferenceC2 = 100
centroidDifferenceC3 = 100


while centroidDifferenceC1 > 0.1 and centroidDifferenceC1 > 0.1 and centroidDifferenceC3 > 0.1:

    previousC1 = c1
    previousC2 = c2
    previousC3 = c3

    dist1 = centroidDistance(xMain, c1)
    dist2 = centroidDistance(xMain, c2)
    dist3 = centroidDistance(xMain, c3)
    dataLabel = labelData(dist1, dist2, dist3)
    c1, c2, c3 = centroidMean(xMain, dataLabel, c1, c2, c3)
    plotData(xA, xB, xC, c1, c2, c3, xMain, dataLabel)

    centroidDifferenceC1 = np.array(math.sqrt((previousC1[0, 0] - c1[0, 0]) ** 2 + (previousC1[0, 1] - c1[0, 1]) ** 2))
    centroidDifferenceC2 = np.array(math.sqrt((previousC2[0, 0] - c2[0, 0]) ** 2 + (previousC2[0, 1] - c2[0, 1]) ** 2))
    centroidDifferenceC3 = np.array(math.sqrt((previousC3[0, 0] - c3[0, 0]) ** 2 + (previousC3[0, 1] - c3[0, 1]) ** 2))


