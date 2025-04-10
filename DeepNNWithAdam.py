import numpy as np
import re
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter

from sklearn.metrics import accuracy_score,classification_report
from sklearn.model_selection import train_test_split



class DeepNetworkWAdam:
    def __init__(self, input_size, hidden_sizes, output_size, learning_rate=0.01, beta=0.9, gamma=0.999, epsilon=0.0001):
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.output_size = output_size
        self.learning_rate = learning_rate
        self.beta = beta   #Adam parameters
        self.gamma = gamma
        self.epsilon = epsilon


        self.W = []  # W is a list that will contain the weight matrices
        self.b = []  # b will contain the biases
        self.m = []  # m and v for Adam
        self.v = []
        self.t = 1  # Step for Adam

        # Initialize weights and bias for first hidden layer
        self.W.append(np.random.randn(input_size, hidden_sizes[0]) * 0.01)
        self.b.append(np.zeros((1, hidden_sizes[0])))

        # Iterate over the hidden layers
        for i in range(1, len(hidden_sizes)):
            #Initialize randomly the weights matrices of the correct size
            self.W.append(np.random.randn(hidden_sizes[i - 1], hidden_sizes[i]) * 0.01)
            #Initialize randomly the bias vectos
            self.b.append(np.zeros((1, hidden_sizes[i])))

        # Last hidden layer to output
        self.W.append(np.random.randn(hidden_sizes[-1], output_size) * 0.01)
        self.b.append(np.zeros((1, output_size)))

        # Adam optimizer parameters for weights and biases
        for i in range(len(self.W)):
            self.m.append([np.zeros_like(self.W[i]), np.zeros_like(self.b[i])])
            self.v.append([np.zeros_like(self.W[i]), np.zeros_like(self.b[i])])

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def sigmoid_derivative(self, z):
        return z * (1 - z)



    def forward(self, X):
        #Keep the activation and preactivation outputs is lists for backpropagation
        self.activations = [X]
        self.preactivations = []

        # Forward step for hidden layers. z are the preacivation outputs. a are the activation outputs
        for i in range(len(self.hidden_sizes)):
            z = np.matmul(self.activations[-1], self.W[i]) + self.b[i]
            a = np.tanh(z)
            self.preactivations.append(z)
            self.activations.append(a)

        # Forward step for output
        z_output = np.matmul(self.activations[-1], self.W[-1]) + self.b[-1]
        output = self.sigmoid(z_output)
        self.preactivations.append(z_output)
        self.activations.append(output)

        return output

    def backward(self, X, y, output):
        output_error = output - y
        output_delta = output_error * self.sigmoid_derivative(output)

        # Gradients for last layer
        dW_last = np.dot(self.activations[-2].T, output_delta)
        db_last = np.sum(output_delta, axis=0, keepdims=True)

        # delta will hold at each step the gradient that is being calculated
        delta = output_delta
        # Lists to hold the gradients
        dW = [dW_last]  #List of weight gradients
        db = [db_last]  #List of bias gradients

        #Moving backwards calculate the weight and bias gradients and insert
        #them at the first positions of dW and db respectively
        for i in range(len(self.hidden_sizes) - 1, -1, -1):
            z = self.preactivations[i]
            activation_gradient = 1 - np.tanh(z) ** 2
            delta = np.matmul(delta, self.W[i + 1].T) * activation_gradient

            #At each step
            dW.insert(0, np.matmul(self.activations[i].T, delta))
            db.insert(0, np.sum(delta, axis=0, keepdims=True))

        return dW, db

    #Method to update parameters with Adam
    def update_params(self, dW, db):
        # Adam update for each layer
        for i in range(len(self.W)):
            # Update  first moment
            self.m[i][0] = self.beta * self.m[i][0] + (1 - self.beta) * dW[i]
            self.m[i][1] = self.beta * self.m[i][1] + (1 - self.beta) * db[i]

            # Update  second  moment
            self.v[i][0] = self.gamma * self.v[i][0] + (1 - self.gamma) * (dW[i] ** 2)
            self.v[i][1] = self.gamma * self.v[i][1] + (1 - self.gamma) * (db[i] ** 2)

            # Correct bias for moment estimates
            m_corrected_W = self.m[i][0] / (1 - self.beta ** self.t)
            v_corrected_W = self.v[i][0] / (1 - self.gamma ** self.t)

            m_corrected_b = self.m[i][1] / (1 - self.beta ** self.t)
            v_corrected_b = self.v[i][1] / (1 - self.gamma ** self.t)

            # Update weights and biases with Adam update rule
            self.W[i] -= self.learning_rate * (m_corrected_W / (np.sqrt(v_corrected_W) + self.epsilon))
            self.b[i] -= self.learning_rate * (m_corrected_b / (np.sqrt(v_corrected_b) + self.epsilon))

        # Increment time step for bias correction
        self.t += 1

    def train(self, X, y, iterations=1000):

        loss_history = []
        for i in range(iterations):
            # Forward pass
            output = self.forward(X)

            # Backward pass
            dW, db = self.backward(X, y, output)

            # Update weights and biases using Adam optimizer
            self.update_params(dW, db)

            # Calculate and store the loss
            if i % 10 == 0:
                loss = np.mean(np.square(y - output))

                loss_history.append(loss)
                if i % 10 == 0:
                    print(f"Iteration {i} - Loss: {loss}")



    def predict(self, X):
        output = self.forward(X)
        return np.round(output)




df=pd.read_csv("C:/Users/yanis/OneDrive/Desktop/amazon.csv")

texts=df['Text'].tolist()         #Extract the reviews column and cast to list() for processing
labels=np.array(df['label'])      #Cast labels to np.array




def preprocess_text(text):          #Clean the text and convert to lowercase
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text) #Keep only alphanumerical characters and whitespaces
    return text

processed_texts = [preprocess_text(text) for text in texts]


all_words = ' '.join(processed_texts).split()
vocab = Counter(all_words)
vocab_size = len(vocab)
word_to_index = {word: idx for idx, (word, _) in enumerate(vocab.items())}

def bow_vector(text, vocab_size, word_to_index):
    vector = np.zeros(vocab_size)
    for word in text.split():
        if word in word_to_index:
            vector[word_to_index[word]] += 1
    return vector

X = np.array([bow_vector(text, vocab_size, word_to_index) for text in processed_texts])


X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.2, random_state=42)
y_train = y_train.reshape(-1, 1)
y_test = y_test.reshape(-1, 1)

nn = DeepNetworkWAdam(input_size=X_train.shape[1],learning_rate=0.01, hidden_sizes=[80,80,50], output_size=1)
nn.train(X_train, y_train, iterations=30)

y_pred = nn.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"Test accuracy: {accuracy * 100:.2f}%")
print(classification_report(y_test,y_pred))
