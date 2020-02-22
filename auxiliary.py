# -*- coding: utf-8 -*-
"""
Created on Tue Feb 11 13:44:37 2020

@author: Team Forest
"""
import numpy as np
import pandas as pd


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
        if data[i, 3] == 'mal':
            data[i, 3] = 0
        elif data[i, 3] == 'fem':
            data[i, 3] = 1
        else:
            data[i, 3] = -1

    return headers, data


def one_out_of_k(input, column_index=0, return_uniques=False):
    """
    Function replacing a specified column of matrix with the One-out-of-K equivalent.
    :param input:
    :param column_index:
    :param return_uniques: 1xN Numpy array, boolean specifying if the unique values of the input should be returned
    :return: The One-out-of-K numpy-matrix and optionally the unique values from the input matrix.
    """
    chosen_column = input[:, column_index]
    uniques = np.unique(chosen_column)

    # creates a matrix of 0's with number of unique entries as columns 
    # and number of input rows as row numbers
    output = np.array(np.zeros(len(uniques) * len(chosen_column))).reshape(len(chosen_column), len(uniques))

    # create a dictionary to lookup the corresponding index of the values in input
    lookup = dict(zip(uniques, np.arange(len(uniques))))

    # insert 1 in correct columns, iterating through the rows
    for i in range(len(chosen_column)):
        j = lookup[chosen_column[i]]
        output[i, j] = 1

    output = np.hstack((np.hstack((input[:, :column_index], output)), input[:, column_index + 1:]))

    # output uniques and output binary matrix in tuple
    if return_uniques:
        return uniques, output
    else:
        return output
