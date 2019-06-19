from keras.datasets import mnist
from keras.utils import to_categorical

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

train_images = train_images.reshape((60000, 28*28,))
train_images = train_images.astype('float32') / 255

test_images = test_images.reshape((10000, 28*28,))
test_images = test_images.astype('float32') / 255

train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)


data_list = [train_images,train_labels,test_images,test_labels]
kar = KARnet(data_list,h=[2000,1000,100],rseed=8)
kar.train()

acc_tr = kar.accuracy(mode='train')
print(acc_tr)
acc_te = kar.accuracy(mode='test')
print(acc_te)

