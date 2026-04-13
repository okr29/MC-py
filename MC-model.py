import pandas
import matplotlib.pyplot as plt
import numpy as np

# header in the data : epsilon_a sigma_a sigma_r epsilon_r epsilon_s p q x x

def load_data(file_path):
    data = pandas.read_csv(file_path, sep=r'\s+')
    return data


def load_other_data(file_path):
    o_data = pandas.read_csv(file_path, sep=r'\s+')
    return o_data

data = load_data('115.dat')
print(data.head())

o_data = load_other_data('triax.out')
print(o_data.head())

# print(data.columns.tolist())

# data.plot(x='epsilon_a', y='q', kind='scatter')
# plt.show()

qmax = data['q'].max()
print(f'Maximum q value: {qmax}')
pmax = data['p'].max()
print(f'Maximum p value: {pmax}')
q_50 = qmax / 2
print(f'Half of maximum q value: {q_50}')

# 1. Find the index (row number) where the distance to q_50 is the absolute minimum
closest_index = (data['q'] - q_50).abs().idxmin()
# 2. Grab the epsilon_a value at that specific row
epsilon_a_q_50 = data.loc[closest_index, 'epsilon_a']
print(f'Closest epsilon_a value: {epsilon_a_q_50}')

E_50 = q_50 / epsilon_a_q_50
print(f'E_50 value: {E_50}')


processed_data = data.assign(epsilon_v=lambda x: x['epsilon_a'] - x['epsilon_r'] * 2)
print(processed_data.head())

processed_odata = o_data.assign(epsilon_v=lambda x: x['epsilon_a'] - x['epsilon_r'] * 2)
print(processed_odata.head())

epsilon_v_min = processed_data['epsilon_v'].min()
print(f'Minimum epsilon_v value: {epsilon_v_min}')

# 1. Find the index (row number) 
closest_index3 = (processed_data['epsilon_v'] - epsilon_v_min).abs().idxmin()
# 2. Grab the value at that specific row
epsilon_a_epsilon_v_min = data.loc[closest_index3, 'epsilon_a']
print(f'Closest epsilon_a value for epsilon_v_min: {epsilon_a_epsilon_v_min}')


nu = epsilon_v_min / epsilon_a_epsilon_v_min
print(f'nu: {nu}')

M = qmax / pmax
print(f'M value: {M}')

sin_phi = 3*M / (6 + M)
phi = np.arcsin(sin_phi)
print(f'Phi: {round(np.degrees(phi), 2)}°')


# 1. Find the index (row number) 
closest_index4 = (processed_data['epsilon_v'] - epsilon_v_min).abs().idxmin()
# 2. Grab the value at that specific row
epsilon_s_epsilon_v_min = data.loc[closest_index4, 'epsilon_s']
print(f'Closest epsilon_s value for epsilon_v_min: {epsilon_s_epsilon_v_min}')

epsilon_v_max = processed_data['epsilon_v'].max()
# print(f'Maximum epsilon_v value: {epsilon_v_max}')
epsilon_v50_od_nuly = (epsilon_v_max - epsilon_v_min) / 2

# 1. Find the index (row number) 
closest_index5 = (processed_data['epsilon_v'] - epsilon_v50_od_nuly).abs().idxmin()
# 2. Grab the value at that specific row
epsilon_s_epsilon_v50_od_nuly = data.loc[closest_index5, 'epsilon_s']
print(f'Closest epsilon_s value for epsilon_v50_od_nuly: {epsilon_s_epsilon_v50_od_nuly}')

M_psi = (epsilon_v50_od_nuly - epsilon_v_min) / (epsilon_s_epsilon_v50_od_nuly - epsilon_s_epsilon_v_min)
print(f'M_psi value: {M_psi}')

sin_psi = 3*M_psi / (6 + M_psi)
psi = np.arcsin(sin_psi)
print(f'Psi: {round(np.degrees(psi), 2)}°')

################################ PLOTOVANI ##########################################
################################ epsilon_a, q ##########################################
plt.scatter(data['epsilon_a'], data['q'], s=2, label='Original Data')
plt.scatter(o_data['epsilon_a'], o_data['q'], s=2, label='Other Data', color='orange')
my_custom_cap = 0.05
x_function = np.linspace(0, my_custom_cap)
y_function = E_50 * x_function
plt.plot(x_function, y_function, label='E50 Secant Line', linestyle='-', color='red')
plt.plot(epsilon_a_q_50, q_50, 'x', markersize=20)
plt.xlabel('epsilon_a')
plt.ylabel('q')
plt.legend()
plt.show()

################################ epsilon_a, epsilon_v ##########################################
################################ nu ##########################################
plt.scatter(processed_data['epsilon_a'], processed_data['epsilon_v'], s=2, label='Original Data')
plt.scatter(processed_odata['epsilon_a'], processed_odata['epsilon_v'], s=2, label='Other Data', color='orange')
plt.plot(np.linspace(0, processed_data['epsilon_a'].max()), np.linspace(0, 0), label='0 Line', linestyle='--', color='red')
my_custom_cap = 0.05
x_function = np.linspace(0, my_custom_cap)
y_function = nu * x_function
plt.plot(x_function, y_function, label='nu Secant Line', linestyle='-', color='red')
plt.xlabel('epsilon_a')
plt.ylabel('epsilon_v')
plt.legend()
plt.show()

################################ epsilon_s, epsilon_v ##########################################
################################ psi ##########################################
plt.scatter(processed_data['epsilon_s'], processed_data['epsilon_v'], s=2, label='Original Data')
plt.plot(np.linspace(0, processed_data['epsilon_s'].max()), np.linspace(0, 0), label='0 Line', linestyle='--', color='red')
plt.scatter(processed_odata['epsilon_s'], processed_odata['epsilon_v'], s=2, label='Other Data', color='orange')
my_custom_cap = 0.35
x_function = np.linspace(0, my_custom_cap)
y_function = M_psi * (x_function - epsilon_s_epsilon_v_min) + epsilon_v_min
plt.plot(x_function, y_function, label='psi Secant Line', linestyle='-', color='red')
plt.xlabel('epsilon_s')
plt.ylabel('epsilon_v')
plt.legend()
plt.show()



################################ p, q ##########################################
plt.scatter(processed_data['p'], processed_data['q'], s=2, label='Original Data')
plt.scatter(processed_odata['p'], processed_odata['q'], s=2, label='Other Data', color='orange')
plt.xlabel('p (kPa)')
plt.ylabel('q (kPa)')
plt.legend()
plt.show()

# ################################ PLOTOVANI ##########################################
# ################################ epsilon_s, q ##########################################
# plt.scatter(data['epsilon_s'], data['q'], label='Original Data')
# my_custom_cap = 0.05
# x_function = np.linspace(0, my_custom_cap)
# y_function = E_50 * x_function
# plt.plot(x_function, y_function, label='E50 Secant Line', linestyle='-', color='red')
# plt.plot(epsilon_a_q_50, q_50, 'x', markersize=20)
# plt.xlabel('epsilon_s')
# plt.ylabel('q')
# plt.legend()
# plt.show()

# # E_50 = q_50 / 