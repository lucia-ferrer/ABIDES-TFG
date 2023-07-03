import pandas as pd
import matplotlib as plt
if __name__ == '__main__':
    results = pd.read_csv(rf"lucia-content\images\images\recovery\{file_name}.csv")
    fig, ax = plt.subplots()
    for unitest in results.columns(): 
       
        id, attack, detector, recover, r_params = unitest.split(':', 5)
        if attack == 'Empty' : 
            empty_attack_col = unitest
            continue
        if recover == 'None': 
            empty_recovery_col = unitest
            continue

        results[unitest].plot(ax=ax, label=f"recovered")
        results[empty_attack_col].plot(ax=ax, label=f"original")
        results[empty_recovery_col].plot(ax=ax, label="attacked")
        ax.legend()
        ax.set_title(unitest)	
        ax.set_ylabel('rewards')	
        ax.set_xlabel('steps')
        plt.save_fig(f"images/recovery/plot_{unitest}.png")
        plt.show()
