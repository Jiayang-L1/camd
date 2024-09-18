import pandas as pd
from sklearn.model_selection import train_test_split


def gofundme_split():
    gofundme = pd.read_pickle('./gofundme_data.pkl')

    train_data, test_data = train_test_split(gofundme, test_size=0.2)
    test_data, val_data = train_test_split(test_data, test_size=0.5)

    train_data = train_data.reset_index(drop=True)
    test_data = test_data.reset_index(drop=True)
    val_data = val_data.reset_index(drop=True)

    train_data.to_pickle('./gofundme_train.pkl')
    test_data.to_pickle('./gofundme_test.pkl')
    val_data.to_pickle('./gofundme_valid.pkl')


def indiegogo_split():
    indiegogo = pd.read_pickle('./indiegogo_data.pkl')

    train_data, test_data = train_test_split(indiegogo, test_size=0.2)
    test_data, val_data = train_test_split(test_data, test_size=0.5)

    train_data = train_data.reset_index(drop=True)
    test_data = test_data.reset_index(drop=True)
    val_data = val_data.reset_index(drop=True)

    train_data.to_pickle('./indiegogo_train.pkl')
    test_data.to_pickle('./indiegogo_test.pkl')
    val_data.to_pickle('./indiegogo_valid.pkl')


if __name__ == '__main__':
    """
    Split datasets into train, validation and test.
    """

    gofundme_split()
    indiegogo_split()
