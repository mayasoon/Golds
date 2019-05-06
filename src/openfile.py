import numpy as np
import matplotlib.pyplot

data_file = open("C:/Users/maya/Desktop/mnist_test_10.csv","r")
data_list = data_file.readlines()
data_file.close()

normal_data = np.asfarray(data_list[0].split(",")[1:]).reshape(1,784)
# print(normal_data)
# two = np.asfarray(data_list[1].split(",")[1:]).reshape(1,784)
# print(two)
# normal_data = np.append(normal_data,two,axis=0)

for i in range(len(data_list)):
    if i > 0:
        arr = np.asfarray(data_list[i].split(',')[1:]).reshape(1,784)
        normal_data = np.append(normal_data,arr,axis=0)
print('normal_data = ',normal_data)


# all_values = data_list[2].split(",")

# image_array = np.asfarray(all_values[1:]).reshape((28,28))
# matplotlib.pyplot.imshow(image_array,cmap="Greys",interpolation="None")
# matplotlib.pyplot.show()

# scaled_input = (np.asfarray(all_values[1:])/255.0 * 0.99) + 0.01
# print(all_values[0])
# onodes = 10
# targets = np.zeros(onodes) + 0.01
# targets[int(all_values[0])] = 0.99
# print(targets)