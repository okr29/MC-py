import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# 1. Define your files here
original_file = '115.dat'
other_files = ['triax.out'] # Add as many comparison files as you want here

# Fixed colors for your primary data
orig_color = '#1f77b4' # Standard matplotlib blue
line_color = 'red'
ideal_mc_color = 'black' # Color for the Idealized MC curve

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
# try:
#     o_data = pd.read_csv(file_path, sep=r'\s+')
#     # Compute the extra columns but nothing else
#     o_data = o_data.assign(epsilon_v=lambda x: x['epsilon_a'] - x['epsilon_r'] * 2)
#     other_datasets.append({'name': file_path, 'data': o_data})
# except FileNotFoundError:
#     print(f"File {file_path} not found. Skipping.")


################################ YIELD PARAMETERS FOR IDEAL MC ######################
# Calculate secant shear modulus G_50
epsilon_s_q_50 = data.loc[closest_index, 'epsilon_s']
G_50 = q_50 / epsilon_s_q_50

# Define the exact coordinates of the Yield Point (the break point)
eps_a_y = qmax / E_50
eps_s_y = qmax / G_50
eps_v_y = nu * eps_a_y

# Secant slope for eps_v vs eps_s in the elastic phase
nu_s = eps_v_y / eps_s_y 

# Derived flow rule slope for eps_v vs eps_a in plastic phase
# Based on the invariant dq/dp = 3 and standard strain mappings
M_psi_a = M_psi / (1 + M_psi / 3) 

# Initial confining pressure (p0) derived from pmax and standard triaxial path
p0 = pmax - (qmax / 3)
#####################################################################################


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

# Shared X-axis arrays for plotting continuous lines
max_eps_a = get_max_val('epsilon_a')
ideal_eps_a = np.linspace(0, max_eps_a, 500)

max_eps_s = get_max_val('epsilon_s')
ideal_eps_s = np.linspace(0, max_eps_s, 500)


# --- Figure 1: epsilon_a vs q ---
plt.figure()
plt.scatter(data['epsilon_a'], data['q'], s=2, label='Original Data', color=orig_color)

my_custom_cap = 0.05
x_function = np.linspace(0, my_custom_cap)
plt.plot(x_function, E_50 * x_function, label='E50 Secant Line', linestyle='-', color=line_color)
plt.plot(epsilon_a_q_50, q_50, 'x', markersize=10, color=line_color)

# TRUE IDEALIZED MC CURVE
ideal_q_mc = np.minimum(E_50 * ideal_eps_a, qmax)
plt.plot(ideal_eps_a, ideal_q_mc, label='Idealized EP-MC', linestyle='--', color=ideal_mc_color, linewidth=2)

for i, ds in enumerate(other_datasets):
    plt.scatter(ds['data']['epsilon_a'], ds['data']['q'], s=2, label=f"{ds['name']}", color=colors(i))

plt.xlabel('epsilon_a')
plt.ylabel('q')
plt.legend()
plt.show()


# --- Figure 2: epsilon_a vs epsilon_v ---
plt.figure()
plt.scatter(data['epsilon_a'], data['epsilon_v'], s=2, label='Original Data', color=orig_color)

my_custom_cap = 0.05
x_function = np.linspace(0, my_custom_cap)
plt.plot(x_function, nu * x_function, label='nu Secant Line', linestyle='-', color=line_color)

# TRUE IDEALIZED MC CURVE (V-shape: Elastic compression then plastic dilation)
ideal_eps_v_a = np.where(ideal_eps_a <= eps_a_y, 
                         nu * ideal_eps_a,  # Elastic phase
                         eps_v_y + M_psi_a * (ideal_eps_a - eps_a_y)) # Plastic phase
plt.plot(ideal_eps_a, ideal_eps_v_a, label='Idealized EP-MC', linestyle='--', color=ideal_mc_color, linewidth=2)

for i, ds in enumerate(other_datasets):
    plt.scatter(ds['data']['epsilon_a'], ds['data']['epsilon_v'], s=2, label=f"{ds['name']}", color=colors(i))

plt.plot(np.linspace(0, max_eps_a), np.linspace(0, 0), label='0 Line', linestyle='--', color='black')
plt.xlabel('epsilon_a')
plt.ylabel('epsilon_v')
plt.legend()
plt.show()


# --- Figure 3: epsilon_s vs epsilon_v ---
plt.figure()
plt.scatter(data['epsilon_s'], data['epsilon_v'], s=2, label='Original Data', color=orig_color)

my_custom_cap = 0.35
x_function = np.linspace(0, my_custom_cap)
y_function = M_psi * (x_function - epsilon_s_epsilon_v_min) + epsilon_v_min
plt.plot(x_function, y_function, label='psi Secant Line', linestyle='-', color=line_color)

# Secant line for original ONLY 80%
my_custom_cap = 0.35
x_function = np.linspace(0, my_custom_cap)
y_function = M_psi_80 * (x_function - epsilon_s_epsilon_v_min) + epsilon_v_min
#plt.plot(x_function, y_function, label='psi_80 Secant Line', linestyle='-', color=line_color)

# Scatter for others
# TRUE IDEALIZED MC CURVE
ideal_eps_v_s = np.where(ideal_eps_s <= eps_s_y, 
                         nu_s * ideal_eps_s, # Elastic phase
                         eps_v_y + M_psi * (ideal_eps_s - eps_s_y)) # Plastic phase
plt.plot(ideal_eps_s, ideal_eps_v_s, label='Idealized EP-MC', linestyle='--', color=ideal_mc_color, linewidth=2)

for i, ds in enumerate(other_datasets):
    plt.scatter(ds['data']['epsilon_s'], ds['data']['epsilon_v'], s=2, label=f"{ds['name']}", color=colors(i))

plt.plot(np.linspace(0, max_eps_s), np.linspace(0, 0), label='0 Line', linestyle='--', color='black')
plt.xlabel('epsilon_s')
plt.ylabel('epsilon_v')
plt.legend()
plt.show()


# --- Figure 4: p vs q ---
plt.figure()
plt.scatter(data['p'], data['q'], s=2, label='Original Data', color=orig_color)

# TRUE IDEALIZED MC CURVE (Stress path straight line ending at failure point)
plt.plot([p0, pmax], [0, qmax], label='Idealized MC Stress Path', linestyle='--', color=ideal_mc_color, linewidth=2)
plt.plot(pmax, qmax, 'ko', markersize=6, label='Yield/Failure Point') # Mark the break point

for i, ds in enumerate(other_datasets):
    plt.scatter(ds['data']['p'], ds['data']['q'], s=2, label=f"{ds['name']}", color=colors(i))

plt.xlabel('p (kPa)')
plt.ylabel('q (kPa)')
plt.legend()
plt.show()


# --- Figure 5: epsilon_s vs q ---
plt.figure()
plt.scatter(data['epsilon_s'], data['q'], s=2, label='Original Data', color=orig_color)

# TRUE IDEALIZED MC CURVE 
ideal_q_s = np.minimum(G_50 * ideal_eps_s, qmax)
plt.plot(ideal_eps_s, ideal_q_s, label='Idealized EP-MC', linestyle='--', color=ideal_mc_color, linewidth=2)

for i, ds in enumerate(other_datasets):
    plt.scatter(ds['data']['epsilon_s'], ds['data']['q'], s=2, label=f"{ds['name']}", color=colors(i))

plt.xlabel('epsilon_s')
plt.ylabel('q')
plt.legend()
plt.show()