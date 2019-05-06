import numpy as np
import matplotlib.pyplot

data_file = open("C:/Users/admin/Desktop/mnist_test_10.csv","r")
data_list = data_file.readlines()
data_file.close()

normal_data = np.array([],dtype=np.float,ndmin=2)

for i in range(len(data_list)):

    arr = data_list[i].split(',')

    print(arr)
print(normal_data)

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