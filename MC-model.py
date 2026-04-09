import pandas

def load_data(file_path):
    data = pandas.read_csv(file_path)
    return data

data = load_data('115.dat')
print(data.head())