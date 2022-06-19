import matplotlib.pyplot as plt
import numpy as np
import csv
import pandas as pd

keyword_map = {
	"sampling": "sampling",
	"features": "aggregating features",
	"training": "training"
}

machine_type_data = {
	"single": "single_node_data",
	"multi": "multi_node_data"
}

machine_type_phrase = {
	"single": "",
	"multi": "2 nodes, "
}

machine_num_gpus = {
	"single": 5,
	"multi": 4
}

def convert():
	sampling = []
	feature = []
	training = []
	epoch = []

	with open('single_node_data/4_gpu_sampling.csv', 'w') as out:
		csv_out = csv.writer(out)
		csv_out.writerow(['total_across_epoch', 'min', 'max'])
		for row in sampling:
			csv_out.writerow(row)

	with open('single_node_data/4_gpu_features.csv', 'w') as out:
		csv_out = csv.writer(out)
		csv_out.writerow(['total_across_epoch', 'min', 'max'])
		for row in feature:
			csv_out.writerow(row)

	with open('single_node_data/4_gpu_training.csv', 'w') as out:
		csv_out = csv.writer(out)
		csv_out.writerow(['total_across_epoch', 'min', 'max'])
		for row in training:
			csv_out.writerow(row)

	x = pd.DataFrame(epoch)
	x.to_csv('single_node_data/4_gpu_epoch.csv', index=False)


def read_csv(file_name, uses_headers=True, usecols=None):
	return pd.read_csv(file_name, header=0 if uses_headers else None, usecols=usecols)


def end_end_stats_composition(machine='single', stat='sampling'):
	root = machine_type_data[machine]
	phrase = machine_type_phrase[machine]

	ax = None
	for i in range(1, machine_num_gpus[machine]):
		df = read_csv(f'{root}/{i}_gpu_{stat}.csv', usecols=['total_across_epoch'])
		ax = df.plot(y='total_across_epoch', kind='line', ax=ax, label=f'{phrase} {i} gpu{""if i == 1 else "s"}')

	ax.set_title(f"Time spent {keyword_map[stat]}")
	ax.set_xlabel("Epoch")
	ax.set_ylabel("Time in s")

	plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left')
	plt.tight_layout()
	plt.show()


def end_end_stats(machine='single'):
	root = machine_type_data[machine]
	phrase = machine_type_phrase[machine]

	ax = None
	for i in range(1, machine_num_gpus[machine]):
		df = read_csv(f'{root}/{i}_gpu_epoch.csv', )
		ax = df.plot(y='0', kind='line', ax=ax, label=f'{phrase} {i} gpu{""if i == 1 else "s"}')

	ax.set_title(f"Time spent in total")
	ax.set_xlabel("Epoch")
	ax.set_ylabel("Time in s")

	plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left')
	plt.tight_layout()
	plt.show()


def percentage_time_spent(machine='single', add_synchronisation=False):

	root = machine_type_data[machine]

	for i in range(1, machine_num_gpus[machine]):
		operations = []
		label_vals = []
		for label in keyword_map.keys():
			df = read_csv(f'{root}/{i}_gpu_{label}.csv', usecols=['total_across_epoch'])
			operations.append(df['total_across_epoch'].sum())
			label_vals.append(keyword_map[label].capitalize())

		if add_synchronisation:
			total_epoch_time = read_csv(f'{root}/{i}_gpu_epoch.csv')

			sync_time = total_epoch_time.sum() - sum(operations)
			operations.append(sync_time[0])
			label_vals.append('GPU synchronisation')

		max_operation = max(operations)
		explode = [0.1 if operation == max_operation else 0.02 for operation in operations]

		fig1, ax1 = plt.subplots()
		ax1.pie(operations,
						explode=explode,
						labels=label_vals,
						autopct='%1.1f%%',
						shadow=True,
						startangle=90)
		ax1.axis('equal')

		ax1.set_title(f"Percentage time spent on each operation using {i} GPUs")
		plt.show()


if __name__ == '__main__':
	end_end_stats_composition(stat='sampling')
	end_end_stats_composition(stat='features')
	end_end_stats_composition(stat='training')
	# TODO Maybe make use of the min/max?
	end_end_stats()

	# percentage_time_spent()
	percentage_time_spent(add_synchronisation=True)

	# Multi Node end-end tests
	machine = 'multi'
	# end_end_stats_composition(machine=machine, stat='sampling')
	# end_end_stats_composition(machine=machine, stat='features')
	# end_end_stats_composition(machine=machine, stat='training')
	# TODO Maybe make use of the min/max?

	# end_end_stats(machine=machine)
	#
	# percentage_time_spent(machine=machine)
	# percentage_time_spent(machine=machine, add_synchronisation=True)
