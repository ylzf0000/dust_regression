import glob

data_train = glob.glob('data_train/*.jpg')
data_valid = glob.glob('data_valid/*.jpg')
data_test = glob.glob('data_test/*.jpg')

print(len(data_train))
print(len(data_valid))
print(len(data_test))