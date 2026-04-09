import pandas
import matplotlib.pyplot as plt
import numpy as np


def load_data(file_path):
    data = pandas.read_csv(file_path, sep=r'\s+')
    return data

data = load_data('115.dat')
print(data.head())

# print(data.columns.tolist())

# data.plot(x='epsilon_a', y='q', kind='scatter')
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




processed_data = data.assign(epsilon_v=lambda x: x['epsilon_a'] - x['epsilon_r'] * 2)
print(processed_data.head())

epsilon_v_min = processed_data['epsilon_v'].min()
print(f'Minimum epsilon_v value: {epsilon_v_min}')

# 1. Find the index (row number) 
closest_index2 = (processed_data['epsilon_v'] - epsilon_v_min).abs().idxmin()
# 2. Grab the value at that specific row
epsilon_a_epsilon_v_min = data.loc[closest_index2, 'epsilon_a']
print(f'Closest epsilon_a value for epsilon_v_min: {epsilon_a_epsilon_v_min}')


# nu = processed_data['epsilon_r'] / processed_data['epsilon_a']
# print(nu.head(50))


################################ PLOTOVANI ##########################################
################################ epsilon_a, q ##########################################
# plt.scatter(data['epsilon_a'], data['q'], label='Original Data')
# plt.plot(x_function, y_function, label='E50 Secant Line', linestyle='-', color='red')
# plt.plot(epsilon_a_q_50, q_50, 'x', markersize=20)
# plt.xlabel('epsilon_a')
# plt.ylabel('q')
# plt.legend()
# plt.show()

################################ epsilon_a, epsilon_v ##########################################
plt.scatter(processed_data['epsilon_a'], processed_data['epsilon_v'], label='Original Data')
plt.plot(np.linspace(0, processed_data['epsilon_a'].max()), np.linspace(0, 0), label='0 Line', linestyle='--', color='red')
plt.xlabel('epsilon_a')
plt.ylabel('epsilon_v')
plt.legend()
plt.show()

# ################################ p, q ##########################################
# plt.scatter(processed_data['p'], processed_data['q'], label='Original Data')
# plt.xlabel('p (kPa)')
# plt.ylabel('q (kPa)')
# plt.legend()
# plt.show()



# E_50 = q_50 / 