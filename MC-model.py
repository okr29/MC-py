import pandas
import matplotlib.pyplot as plt
import numpy as np


def load_data(file_path):
    data = pandas.read_csv(file_path, sep=r'\s+')
    return data

data = load_data('115.dat')
print(data.head())

# print(data.columns.tolist())

data.plot(x='epsilon_a', y='q', kind='scatter')
# plt.show()

qmax = data['q'].max()
print(f'Maximum q value: {qmax}')
q_50 = qmax / 2
print(f'Half of maximum q value: {q_50}')

# 1. Find the index (row number) where the distance to q_50 is the absolute minimum
closest_index = (data['q'] - q_50).abs().idxmin()
# 2. Grab the epsilon_a value at that specific row
epsilon_a_q_50 = data.loc[closest_index, 'epsilon_a']
print(f'Closest epsilon_a value: {epsilon_a_q_50}')

E_50 = q_50 / epsilon_a_q_50
my_custom_cap = 0.05
x_function = np.linspace(0, my_custom_cap)
y_function = E_50 * x_function

plt.plot(data['epsilon_a'], data['q'], label='Original Data')
plt.plot(x_function, y_function, label='E50 Secant Line', linestyle='-', color='red')
plt.plot(epsilon_a_q_50, q_50, 'x', markersize=20)
plt.xlabel('epsilon_a')
plt.ylabel('q')
plt.legend()
plt.show()


# E_50 = q_50 / 