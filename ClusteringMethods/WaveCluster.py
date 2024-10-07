# Оригинальный исходный код взят из https://github.com/jimmybow/waveCluster/blob/358c0945e9b9cc58711fb2dbb31c58cebd6a9108/waveCluster.py
# и изначально написан jimmybow под лицензией MIT

# TODO: Fix type-hinting

import matplotlib.pyplot as plt
import numpy as np
from math import (
    ceil
)

from typing import (
    Dict,
    List,
    Set,
    Tuple
)
from numpy.typing import NDArray


# calculate 1 order n dimensional wavelet transform with numbered grids
KnownWavelets = {
    'db1': [0.707, 0.707],
    'db2': [-0.13, 0.224, 0.836, 0.483],
    'bior1.3': [-0.09, 0.09, 0.707, 0.707, 0.09, -0.09]
}

# the input data can only have one more row represent tags at the end
KnownWaveletsLengths = {
    'db1': 0,
    'db2': 1,
    'bior1.3': 2
}


def scale_01_data(rawData: NDArray[np.number]) -> NDArray[np.number]:
    # normalize the raw dataset
    dim = rawData.shape[1]

    # the rawData has at least 2 raw, 1 for signal 1 for lable
    minList = [np.amin(rawData[:,x]) for x in range(0, dim)]
    maxList = [np.amax(rawData[:,x])+0.001 for x in range(0, dim)] 

    # add the [0] and [1] because there is a 'raw of lable', and 0.001 to avoid 1
    toZero = rawData - np.array(minList)
    normData = toZero / (np.array(maxList) - np.array(minList))

    return normData

def map2ScaleDomain(dataset: NDArray[np.number], scale = 128) -> Dict[int, int]:
    # map the dataset into scale domain for wavelet transform
    if scale <= 0 or not(isinstance(scale, int)):
        raise ValueError('scale must be a positive interger')

    dim = dataset.shape[1]
    length = dataset.shape[0]
    sd_data: Dict[int, int] = {}

    for i in range(0,length):
        num = 0
        for j in reversed(range(0, dim)):     # start from the most weighted dimension
            num += (dataset[i,j]//(1/scale))*pow(scale, j)  # let the numbering start from '0'!
        num = int(num)
        if num not in sd_data:
            sd_data[num] = 1
        else:
            sd_data[num] += 1

    return sd_data

def ndWT(data: Dict[int, int]|Dict[int, float], dim: int, scale: int, wave: str) -> Dict[int, int]|Dict[int, float]:
    lowFreq: Dict[int, float] = {}

    wavelet = KnownWavelets.get(wave)
    if wavelet is None:
        raise ValueError(f"unknown wave {wave} requested")

    convolutionLen = len(wavelet) - 1
    lineLen = ceil(scale / 2) + ceil((convolutionLen - 2) / 2)

    for inDim in range(0, dim):
        for key in data.keys():
            coordinate: List[int] = [] # coordinate start from 0
            tempkey = key
            for i in range(0,dim):
                # get the coordinate for a numbered grid
                if i <= dim-inDim - 1:
                    coordinate.append(tempkey // pow(scale, (dim - 1 - i)))
                    tempkey = tempkey % pow(scale, (dim - 1 - i))
                else:
                    coordinate.append(tempkey // pow(lineLen, (dim - 1 - i)))
                    tempkey = tempkey % pow(lineLen, (dim - 1 - i))

            coordinate.reverse()

            # to calculate ndwt, signal should start from 1, temperally convert
            startCoord = ceil((coordinate[inDim] + 1) / 2) - 1
            startNum = 0    # numbered lable for next level of data
            for i in range(0, dim):
                if i <= inDim:
                    if i == inDim:
                        startNum += startCoord * pow(lineLen, i)
                    else:
                        startNum += coordinate[i] * pow(lineLen, i)
                else:
                    startNum += coordinate[i] * pow(scale, i)

            for i in range(0, convolutionLen // 2 + 1):  
                if startCoord+i >= lineLen: # coordinate start from 0 
                    break

                idx = int(startNum+pow(lineLen, inDim)*i)
                if idx not in lowFreq:
                    lowFreq[idx] = \
                            data[key]*wavelet[int((startCoord+1+i)*2-(coordinate[inDim]+1))]
                else:
                    lowFreq[idx] += \
                            data[key]*wavelet[int((startCoord+1+i)*2-(coordinate[inDim]+1))]

        data = lowFreq
        lowFreq = {}

    return data

# start node checking
class node():
    def __init__(self, key = 0, value = 0):
        self.key = key
        self.value = value
        self.process = False
        self.cluster: None|int = None

    def around(self, scale = 1, dim = 1) -> List[int]:
        aroundNodeKey: List[int] = []

        for inDim in range(0, dim):
            # we can't afford diagnal searching
            dimCoord: int = self.key // pow(scale, inDim)
            if dimCoord == 0:
                aroundNodeKey.append(self.key + pow(scale, inDim))
            elif dimCoord == scale - 1:
                aroundNodeKey.append(self.key - pow(scale, inDim))
            else:
                aroundNodeKey.append(self.key + pow(scale, inDim))
                aroundNodeKey.append(self.key - pow(scale, inDim))

        return aroundNodeKey

def bfs(equal_pair: Set[Tuple[int, int]], maxQueue: int) -> List[List[int]]:
    if len(equal_pair) == 0:
        return []

    group: Dict[int, List[int]] = {x:[] for x in range(1, maxQueue)}
    result: List[List[int]] = []

    for x,y in equal_pair:
        group[x].append(y)
        group[y].append(x)

    for i in range(1, maxQueue):
        if i in group:
            if len(group[i]) == 0:
                del group[i]
            else:
                queue = [i]
                for j in queue:
                    if j in group:
                        queue += group[j]
                        del group[j]
                record = list(set(queue))
                record.sort()
                result.append(record)

    return result

def build_key_cluster(nodes: Dict[int, node], equal_list: List[List[int]], cutMiniCluster: float) -> Dict[int, int]:
    cluster_key: Dict[int, List[node]] = {}

    for point in nodes.values():
        flag = 0
        for cluster in equal_list:
            if point.cluster in cluster:
                point.cluster = cluster[0]
                if cluster[0] not in cluster_key:
                    cluster_key[cluster[0]] = [point]
                    flag = 1
                else:
                    cluster_key[cluster[0]].append(point)
                    flag = 1
                break
        if flag == 0:
            if point.cluster is None:
                continue

            if point.cluster not in cluster_key:
                cluster_key[point.cluster] = [point]
            else:
                cluster_key[point.cluster].append(point)

    count = 1
    result: Dict[int, int] = {}

    for cluster in cluster_key.keys():
        if len(cluster_key[cluster]) == 1:
            if cluster_key[cluster][0].value < cutMiniCluster:
                continue

        for p in cluster_key[cluster]:
            result[p.key] = count

        count += 1

    return result

def clustering(data: Dict[int, node], scale: int, dim: int, cutMiniCluster: float) -> Dict[int, int]:
    equal_pair: List[Tuple[int, int]]|Set[Tuple[int, int]] = []
    cluster_flag = 1

    for point in data.values():
        point.process = True
        for around in point.around(scale, dim):
            aroundNode = data.get(around)
            if aroundNode is None:
                continue

            if aroundNode.cluster is not None:
                if point.cluster is None:
                    point.cluster = aroundNode.cluster
                elif point.cluster != aroundNode.cluster:
                    mincluster = min(point.cluster, aroundNode.cluster)
                    maxcluster = max(point.cluster, aroundNode.cluster)
                    equal_pair += [(mincluster,maxcluster)]

        if point.cluster is None:
            point.cluster = cluster_flag
            cluster_flag += 1

    equal_pair = set(equal_pair)
    equal_list = bfs(equal_pair, cluster_flag)

    return build_key_cluster(data,equal_list,cutMiniCluster)

def thresholding(data: Dict[int, int]|Dict[int, float], threshold: float, scale: int, dim: int) -> Dict[int, int]:
    nodes: Dict[int, node] = {}
    startNode = node(0)
    avg = 0

    for key,value in data.items():
        if value >= threshold:
            nodes[key] = node(key, value)
            avg += value
            if value > startNode.value:
                startNode = node(key, value)

    cutMiniCluster = avg / len(nodes)

    return clustering(nodes,scale,dim,cutMiniCluster)

def findThreshold(data: Dict[int, int]|Dict[int, float], threshold: float):
    value = list(data.values())
    value.sort(reverse=True)

    # 'cutMiniCluster' is used to throw away the single grid
    x = [i for i in range(1, len(value)+1)]

    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.scatter(x,value)
    ax.axhline(y=threshold,xmin=0,xmax=1,color='r')
    plt.show()

def markData(normData: NDArray[np.float64], cluster: Dict[int, int], scale: int) -> List[int]:
    dim = normData.shape[1]

    # there is a column for tags
    tags: List[int] = []
    for point in range(0,normData.shape[0]):
        number = 0
        for inDim in range(0,dim):
            number += (normData[point,inDim] // (1 / scale)) * pow(scale,inDim)

        neededClusterVal = cluster.get(int(number))
        if neededClusterVal is None:
            tags.append(0)
        else:
            tags.append(neededClusterVal)

    return tags

def waveCluster(data: NDArray[np.number], scale = 50, wavelet = 'db2', threshold = 0.5, plot = False) -> List[int]:
    normData = scale_01_data(data)
    dim = normData.shape[1]
    dataDic = map2ScaleDomain(normData,scale)
    dwtResult = ndWT(dataDic,dim,scale,wavelet)

    if plot:
        findThreshold(dwtResult, threshold)

    selectedWaveletLen = KnownWaveletsLengths.get(wavelet)
    if selectedWaveletLen is None:
        raise ValueError(f"wavelet {wavelet} not found")

    lineLen = scale // 2 + selectedWaveletLen
    result = thresholding(dwtResult, threshold, lineLen, dim)

    return markData(normData, result, lineLen)
