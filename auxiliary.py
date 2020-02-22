# -*- coding: utf-8 -*-
"""
Created on Tue Feb 11 13:44:37 2020

@author: Team Forest
"""
import numpy as np
import pandas as pd

""""""
def load_csv(path):
    """
    Reads a file (csv) and formats it as specified.
    :param path:
    :return: Tuple containing numpy array of headers and data values respectively.
    """
    data = pd.read_csv(path)
    headers = data.columns.values
    data = data.values

    # formats gender to numerical, "boolean" value
    for i in range(len(data)):
        if data[i,3] == 'mal':
            data[i,3] = 0
        elif data[i,3] == 'fem':
            data[i,3] = 1
        else:
            data[i,3] = -1

    return headers, data


def one_out_of_k(input, return_uniques=False):
    """
    Function returning the One-out-of-K of a given array.
    :param input, return_uniques: 1xN Numpy array, boolean specifying if the unique values of the input should be returned
    :return: The One-out-of-K numpy-matrix and optionally the unique values from the input array.
    """
    uniques = np.unique(input)

    # creates a matrix of 0's with number of unique entries as columns 
    # and number of input rows as row numbers
    output = np.array(np.zeros(len(uniques) * len(input))).reshape(len(input), len(uniques))

    # create a dictionary to lookup the corresponding index of the values in input
    lookup = dict(zip(uniques, np.arange(len(uniques))))

    # insert 1 in correct columns, iterating through the rows
    for i in range(len(input)):
        j = lookup[input[i]]
        output[i, j] = 1

    # output uniques and output binary matrix in tuple
    if return_uniques:
        return uniques, output
    else:
        return output

path = "./res/bone_density.csv"
headers, data = load_csv(path)
print(headers)
print(data)