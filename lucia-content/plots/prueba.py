import pandas as pd
import matplotlib.pyplot as plt
import os
if __name__ == '__main__':
	
	parent = os.getcwd()[:-6] 
	file_name = parent + "/images/recovery/epsidoes_img_recovery_test__DBSCAN_UniformAttack_2023-07-02.csv"
	results = pd.read_csv(file_name, header=0)
	fig, ax = plt.subplots()
	fig.figsize = (10,10)
	recovers = []
	for unitest in results.columns:
		print(unitest)    
		id, attack, detector, recover = unitest.split(':', maxsplit=3)
		recover, r_params = recover.split(', \\', 1)
		if attack == 'Empty' : 
			empty_attack_col = unitest
			continue
		if recover == 'None': 
			empty_recovery_col = unitest
			continue
		recovers.append(unitest)
	
	for unitest in recovers: 
		l1 = results[unitest].plot(ax=ax, label=f"recovered")
		l2 = results[empty_attack_col].plot(ax=ax, label=f"original")
		l3 = results[empty_recovery_col].plot(ax=ax, label="attacked")

		ax.set_title(unitest)	
		ax.set_ylabel('rewards')	
		ax.set_xlabel('steps')
		ax.set_yscale('symlog')
		for line, style, color in zip(ax.get_lines(), ['dashed', 'solid', 'dotted'], ['green', 'black', 'red']):#'dashdot'
			line.set_linestyle(style)
			line.color = color
		ax.legend()
		plt.savefig(parent + f"/images/recovery/plot_{unitest}.png")
		ax.clear()
		
