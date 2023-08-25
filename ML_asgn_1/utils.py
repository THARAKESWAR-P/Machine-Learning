import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


def get_feature_names(path):
    '''
        Gets the column names from the given path
    '''
    data = pd.read_csv(path)
    return data.columns


