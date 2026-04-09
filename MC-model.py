import pandas
import matplotlib.pyplot as plt

def load_data(file_path):
    data = pandas.read_csv(file_path, sep=r'\s+')
    return data

data = load_data('115.dat')
print(data.head())

# print(data.columns.tolist())

data.plot(x='epsilon_a', y='q', kind='scatter')
plt.show()