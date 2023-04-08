import pandas as pd
import numpy as np

if __name__ == '__main__':
    
    ids = ['MM', 'PT1']
    #transitions = {id: pd.read_csv(f'data/transitions_{id}.csv', header=0).values for id in ids}
    transitions = {id: pd.read_csv(f'transitions_{id}.csv', header=0).values for id in ids}
    state_dimns = {'MM': 28, 'PT1':21}
    for id in ids: 
        transitions[id]['ITER'] = transitions[id].index
        transitions[id]['Date'] = (pd.date_range(start=pd.datetime.datetime(2023, 3, 1), 
                                     periods=transitions[id].shape[0], 
                                     freq='30S'))

        print(f'Shape of data {transitions[id].shape}')
        for i in range(state_dimns[id]):
            transitions[id][i].plot(figsize=(12,5))



