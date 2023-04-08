import pandas as pd
import numpy as np

if __name__ == '__main__':
    
    ids = ['MM', 'PT1']
    #transitions = {id: pd.read_csv(f'data/transitions_{id}.csv', header=0).values for id in ids}
    transitions = {id: pd.read_csv(f'transitions_{id}.csv', header=0).values for id in ids}
    state_dimns = {'MM': 28, 'PT1':21}
    for id, data in transitions.items: 
        data['Date'] = (pd.date_range(start=pd.datetime.datetime(2023, 3, 1), 
                                     periods=data.shape[0], 
                                     freq='30S'))
        data.set_index('Date')

        print(f'Shape of data for {id} -> {transitions[id].shape}')
        for i in range(state_dimns[id]):
            data[i].plot(figsize=(12,5))



