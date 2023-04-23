import numpy as np
from typing import List,Tuple
def concatLists(*lists:list)->list:
    res=[]
    for i in lists:
        res.extend(i)
    return res
def meanOfPairs(pairs:List[Tuple])->Tuple:
    arr=np.array(pairs)
    arr=arr.T
    mean=[0]*len(pairs[0])
    for i in range(len(arr)):
        mean[i]=np.mean(arr[i])
    return tuple(mean)
