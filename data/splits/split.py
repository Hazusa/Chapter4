from sklearn.model_selection import train_test_split

from data.raw.trans import npy_data, labels

X_train, X_val, y_train, y_val = train_test_split(npy_data, labels, test_size=0.2)