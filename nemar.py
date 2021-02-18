import pickle
import pandas as pd

if __name__ == '__main__':
    file_name = './' + 'output_evalita_100_nemar/' + 'nemar_29999.pkl'
    with open(file_name, "rb") as fp:
        data = pickle.load(fp)

    df_evalita = pd.DataFrame.from_dict(data, orient='index')




