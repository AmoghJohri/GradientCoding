from sklearn import datasets

def get_dataset(s="boston"):
    if   s == "boston":       
        return datasets.load_boston()
    elif s == "iris":
        return datasets.load_iris()
    elif s == "diabetes":
        return datasets.load_diabetes()
    elif s == "digits":
        return datasets.load_digits()
    elif s == "linnerud":
        return datasets.load_linnerud()
    elif s == "wine":
        return datasets.load_wine()
    elif s == "breast_cancer":
        return datasets.load_breast_cancer()
    else:
        print("Invalid Option!")

if __name__ == "__main__":
    data = get_dataset("iris")
    X, y = data.data, data.target
    print(y)