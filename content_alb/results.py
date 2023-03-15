import numpy as np
import pandas as pd


def ttest(a, b):
    return (a[0]-b[0])/np.sqrt(a[1]**2+b[1]**2)


og = [-9297,  62641]
attacks = {
    'UniformAttack': [-77380, 10610],
    'STAttack': [-55104, 7679],
    'VFAttack': [-94171,  14322],
    'Empty': []
}

df = pd.read_csv('results/recovery.csv')
df = df[df.norm == 'z-score']
for (attack, detector), d in iter(df.groupby(['attack', 'detector'])):
    if attacks[attack]:
        print(attack, detector)
        ta = ttest(og, attacks[attack])
        # print(ta, og, attacks[attack])
        for _, r in d.iterrows():
            t = ttest(og, r[['recovered_reward','recovered_reward_std']])
            # print(r['recovery_params'], (t-ta)/-ta)
            print((t-ta)/-ta)
            # print(r[['recovered_reward','recovered_reward_std']])
     # [['recovered_reward','recovered_reward_std', 'recovery_params']]