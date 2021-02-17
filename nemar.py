import pickle

if __name__ == '__main__':
    file_name = './' + 'output_2021-02-17-23:54:42/' + 'nemar_0.pkl'
    with open(file_name, "rb") as fp:
        data = pickle.load(fp)
    print(data)
