import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

# Binds an input in [-inf, +inf] to [0, 1]
def sigmoid(x):
    # Activation function:
    return 1 / (1 + np.exp(-x))


# Returns the derivative of the sigmoid function
def ddx_sigmoid(x):
  # Derivative of sigmoid: f'(x) = f(x) * (1 - f(x))
  fx = sigmoid(x)
  return fx * (1 - fx)

"""
Numpy arrays will substract the ith element of pred from true
len(y_true) == len(y_pred)
returns the mean squared average of the outputs (loss)
"""
def mse(y_true, y_pred):
    return ((y_true - y_pred) ** 2).mean()

class NeuralNetwork:
    """
    A neural network with:
        - 28 * 28 inputs
        - 1 bias neuron
        - 1 hidden layer with 20 neurons
        - an output layer with 10 neurons
    """

    # Initializes all weights and biases to random values
    def __init__(self):
        """
        Bias neuron with all 0's as weights
        Want to start with an UNBIASED network
        Bias neuron shifts the function vertically
        """
        self.b_i_h = np.zeros((20, 1))
        self.b_h_o = np.zeros((10, 1))
        
        # 28 * 28 = 784 = num of pixels in each image
        self.w_i_h = np.random.uniform(-0.5, 0.5, (20, 784))
        self.w_h_o = np.random.uniform(-0.5, 0.5, (10, 20))

        """
        Images has a shape of 60000 x 784
        Labels has a shape of 60000 x 10 (one-hot encoding)
            Each value out of the 10 maps to a categorical representation of 0-9 (digits in the dataset)
            0 = NOT PRESENT
            1 = PRESENT
        """

        # Load dataset
        (images, labels), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

        # One-hot encode labels
        self.labels = tf.keras.utils.to_categorical(labels, num_classes=10)

        # Reshape images from (60000, 28, 28) to (60000, 784)
        self.images = images.reshape(60000, 784)

        self.learn_rate = 0.001
        self.nr_correct = 0
        self.epochs = 10

    def train(self):
        print("Training network...")
        for epoch in range(self.epochs):
            for img, l in zip(self.images, self.labels):
                # Transforms the vector into a 1 x 784 matrix
                img.shape += (1,)
                # Transforms the vector into a 1 x 10 matrix
                l.shape += (1,)

                """
                Forward propogation (i -> h)
                Bind input between [0,1]
                Add the bias matrix to the product of the img matrix and the weight matrix
                """
                h_pre = self.b_i_h + self.w_i_h @ img
                h = sigmoid(h_pre)

                # Forward propagation hidden -> output
                o_pre = self.b_h_o + self.w_h_o @ h
                o = 1 / (1 + np.exp(-o_pre))

                # Compare output values with the label via the cost/error function
                error = mse(l, o)
                
                """
                Returns 0 if the output incorrectly produces the label
                Returns 1 if the output correctly matches the label
                """
                self.nr_correct += int(np.argmax(o) == np.argmax(l))


                """
                Backpropogation of the output to the hidden layer
                The goal is to MINIMIZE the ERROR
                """
                delta_o = o - l # Difference between the output and label matrices
                self.w_h_o += -self.learn_rate * delta_o @ np.transpose(h) # Updates the weights of h -> o
                self.b_h_o += -self.learn_rate * delta_o

                """
                Backpropogation of the hidden layer to the inputs
                """
                delta_h = np.transpose(self.w_h_o) @ delta_o * ddx_sigmoid(h)
                self.w_i_h += -self.learn_rate * delta_h @ np.transpose(img) 
                self.b_i_h += -self.learn_rate * delta_h

            # Show accuracy for this epoch
            print(f"Acc: {round((self.nr_correct / self.images.shape[0]) * 100, 2)}%")
            self.nr_correct = 0

    def use(self):
        index = int(input("Enter a number (0 - 59999): "))
        img = self.images[index]
        plt.imshow(img.reshape(28, 28), cmap="Greys")

        img.shape += (1,)
        # Forward propagation input -> hidden
        h_pre = self.b_i_h + self.w_i_h @ img.reshape(784, 1)
        h = sigmoid(h_pre)
        # Forward propagation hidden -> output
        o_pre = self.b_h_o + self.w_h_o @ h
        o = sigmoid(o_pre)
        plt.title(f"Prediction is: {o.argmax()}")
        plt.show()

network = NeuralNetwork()
network.train()
network.use()

    








