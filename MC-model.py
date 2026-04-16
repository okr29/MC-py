import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# 1. Define your files here
original_file = '115.dat'
other_files = ['triax.out', 'triax2.out'] # Add as many comparison files as you want here

# Fixed colors for your primary data
orig_color = '#1f77b4' # Standard matplotlib blue
line_color = 'red'

# Define the headers for the original data
og_headers = ['epsilon_a', 'sigma_a', 'sigma_r', 'epsilon_r', 'epsilon_s', 'p', 'q', 'x1', 'x2', 'x3']

print(f"--- Calculating complex values for ORIGINAL DATA: {original_file} ---")

# --- PROCESS ORIGINAL DATA ---
# Read CSV with header=None and apply the custom column names
data = pd.read_csv(original_file, sep=r'\s+', header=None, names=og_headers)
data = data.assign(epsilon_v=lambda x: x['epsilon_a'] - x['epsilon_r'] * 2)

qmax = data['q'].max()
pmax = data['p'].max()
q_50 = qmax / 2

# E_50
closest_index = (data['q'] - q_50).abs().idxmin()
epsilon_a_q_50 = data.loc[closest_index, 'epsilon_a']
E_50 = q_50 / epsilon_a_q_50

# nu
epsilon_v_min = data['epsilon_v'].min()
closest_index3 = (data['epsilon_v'] - epsilon_v_min).abs().idxmin()
epsilon_a_epsilon_v_min = data.loc[closest_index3, 'epsilon_a']
nu = epsilon_v_min / epsilon_a_epsilon_v_min

# phi
M = qmax / pmax
sin_phi = 3 * M / (6 + M)
phi = np.arcsin(sin_phi)

# psi
closest_index4 = (data['epsilon_v'] - epsilon_v_min).abs().idxmin()
epsilon_s_epsilon_v_min = data.loc[closest_index4, 'epsilon_s']

epsilon_v_max = data['epsilon_v'].max()
epsilon_v50_od_nuly = (epsilon_v_max - epsilon_v_min) / 2
epsilon_v80_od_nuly = (epsilon_v_max - epsilon_v_min) * 0.8

closest_index6 = (data['epsilon_v'] - epsilon_v50_od_nuly).abs().idxmin()
epsilon_s_epsilon_v50_od_nuly = data.loc[closest_index6, 'epsilon_s']

closest_index6 = (data['epsilon_v'] - epsilon_v80_od_nuly).abs().idxmin()
epsilon_s_epsilon_v80_od_nuly = data.loc[closest_index6, 'epsilon_s']

M_psi = (epsilon_v50_od_nuly - epsilon_v_min) / (epsilon_s_epsilon_v50_od_nuly - epsilon_s_epsilon_v_min)
M_psi_80 = (epsilon_v80_od_nuly - epsilon_v_min) / (epsilon_s_epsilon_v80_od_nuly - epsilon_s_epsilon_v_min)
sin_psi = 3 * M_psi / (6 + M_psi)
sin_psi_80 = 3 * M_psi_80 / (6 + M_psi_80)
psi = np.arcsin(sin_psi)
psi_80 = np.arcsin(sin_psi_80)

print(f"E_50: {E_50:.2f} | nu: {nu:.4f} | Phi: {np.degrees(phi):.2f}° | Psi: {np.degrees(psi):.2f}° | Psi_80: {np.degrees(psi_80):.2f}°\n")


# Define the headers for the .out files (9 columns)
out_headers = ['epsilon_a', 'sigma_a', 'sigma_r', 'epsilon_r', 'epsilon_s', 'p', 'q', 'x1', 'x2']

# --- PROCESS OTHER DATA ---
other_datasets = []
for file_path in other_files:
    print(f"--- Loading basic data for: {file_path} ---")
    
    # Add header=None and names=out_headers here!
    o_data = pd.read_csv(file_path, sep=r'\s+', header=None, names=out_headers)
    
    # Compute the extra columns but nothing else
    o_data = o_data.assign(epsilon_v=lambda x: x['epsilon_a'] - x['epsilon_r'] * 2)
    
    other_datasets.append({
        'name': file_path,
        'data': o_data
    })


################################ PLOTOVANI ##########################################

# Color map for the "other" files
colors = plt.colormaps['Set3']

# Helper to find max X values for drawing 0-lines across all datasets
def get_max_val(column_name):
    max_orig = data[column_name].max()
    if not other_datasets:
        return max_orig
    max_others = max(ds['data'][column_name].max() for ds in other_datasets)
    return max(max_orig, max_others)

# --- Figure 1: epsilon_a vs q ---
plt.figure()
plt.scatter(data['epsilon_a'], data['q'], s=2, label='Original Data', color=orig_color)

# Secant line for original ONLY
my_custom_cap = 0.05
x_function = np.linspace(0, my_custom_cap)
y_function = E_50 * x_function
plt.plot(x_function, y_function, label='E50 Secant Line', linestyle='-', color=line_color)
plt.plot(epsilon_a_q_50, q_50, 'x', markersize=10, color=line_color)

# Scatter for others
for i, ds in enumerate(other_datasets):
    plt.scatter(ds['data']['epsilon_a'], ds['data']['q'], s=2, label=f"{ds['name']}", color=colors(i))

plt.xlabel('epsilon_a')
plt.ylabel('q')
plt.legend()
plt.show()

# --- Figure 2: epsilon_a vs epsilon_v ---
plt.figure()
plt.scatter(data['epsilon_a'], data['epsilon_v'], s=2, label='Original Data', color=orig_color)

# Secant line for original ONLY
my_custom_cap = 0.05
x_function = np.linspace(0, my_custom_cap)
y_function = nu * x_function
plt.plot(x_function, y_function, label='nu Secant Line', linestyle='-', color=line_color)

# Scatter for others
for i, ds in enumerate(other_datasets):
    plt.scatter(ds['data']['epsilon_a'], ds['data']['epsilon_v'], s=2, label=f"{ds['name']}", color=colors(i))

# 0 Line
plt.plot(np.linspace(0, get_max_val('epsilon_a')), np.linspace(0, 0), label='0 Line', linestyle='--', color='black')

plt.xlabel('epsilon_a')
plt.ylabel('epsilon_v')
plt.legend()
plt.show()

# --- Figure 3: epsilon_s vs epsilon_v ---
plt.figure()
plt.scatter(data['epsilon_s'], data['epsilon_v'], s=2, label='Original Data', color=orig_color)

# Secant line for original ONLY
my_custom_cap = 0.35
x_function = np.linspace(0, my_custom_cap)
y_function = M_psi * (x_function - epsilon_s_epsilon_v_min) + epsilon_v_min
plt.plot(x_function, y_function, label='psi Secant Line', linestyle='-', color=line_color)

# Secant line for original ONLY 80%
my_custom_cap = 0.35
x_function = np.linspace(0, my_custom_cap)
y_function = M_psi_80 * (x_function - epsilon_s_epsilon_v_min) + epsilon_v_min
plt.plot(x_function, y_function, label='psi_80 Secant Line', linestyle='-', color=line_color)

# Scatter for others
for i, ds in enumerate(other_datasets):
    plt.scatter(ds['data']['epsilon_s'], ds['data']['epsilon_v'], s=2, label=f"{ds['name']}", color=colors(i))

# 0 Line
plt.plot(np.linspace(0, get_max_val('epsilon_s')), np.linspace(0, 0), label='0 Line', linestyle='--', color='black')

plt.xlabel('epsilon_s')
plt.ylabel('epsilon_v')
plt.legend()
plt.show()

# --- Figure 4: p vs q ---
plt.figure()
plt.scatter(data['p'], data['q'], s=2, label='Original Data', color=orig_color)

# Scatter for others
for i, ds in enumerate(other_datasets):
    plt.scatter(ds['data']['p'], ds['data']['q'], s=2, label=f"{ds['name']}", color=colors(i))

plt.xlabel('p (kPa)')
plt.ylabel('q (kPa)')
plt.legend()
plt.show()


# --- Figure 5: epsilon_s vs q ---
plt.figure()
plt.scatter(data['epsilon_s'], data['q'], s=2, label='Original Data', color=orig_color)

# Scatter for others
for i, ds in enumerate(other_datasets):
    plt.scatter(ds['data']['epsilon_s'], ds['data']['q'], s=2, label=f"{ds['name']}", color=colors(i))

plt.xlabel('epsilon_s')
plt.ylabel('q')
plt.legend()
plt.show()
