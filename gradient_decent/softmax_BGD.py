import numpy as np

class SoftmaxRegression:
    def __init__(self, learning_rate=0.01, n_iter=1000):
        self.learning_rate = learning_rate
        self.n_iter = n_iter
        self.weights = None
        self.bias = None
        self.training_losses = []
        self.validation_losses = []
        self.training_accuracies = []
        self.validation_accuracies = []

    def softmax(self, z):
        exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)

    def one_hot_encode(self, y, n_classes):
        return np.eye(n_classes)[y]

    def fit(self, X, y, X_val, y_val):
        n_samples, n_features = X.shape
        n_classes = len(np.unique(y))

        self.weights = np.zeros((n_features, n_classes))
        self.bias = np.zeros(n_classes)

        for i in range(self.n_iter):
            # Tính toán dự đoán cho tập train
            linear_model = np.dot(X, self.weights) + self.bias
            y_predicted = self.softmax(linear_model)

            # Tính toán training loss 
            training_loss = self._compute_loss(X, y)
            self.training_losses.append(training_loss)

            # Cập nhật trọng số và độ dời
            dw = (1 / n_samples) * np.dot(X.T, (y_predicted - self.one_hot_encode(y, n_classes)))
            db = (1 / n_samples) * np.sum(y_predicted - self.one_hot_encode(y, n_classes), axis=0)
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

            # Tính toán accuracy trên tập training
            training_accuracy = self._compute_accuracy(X, y)
            self.training_accuracies.append(training_accuracy)

            # Tính toán loss và accuracy trên tập validation
            val_loss = self._compute_loss(X_val, y_val)
            val_accuracy = self._compute_accuracy(X_val, y_val)
            self.validation_losses.append(val_loss)
            self.validation_accuracies.append(val_accuracy)

            # In thông tin mỗi 100 epochs (thay vì 1000 epochs)
            if i % 1000 == 0:
                print(f"Epoch {i}, Training Loss: {training_loss:.4f}, Validation Loss: {val_loss:.4f}, "
                      f"Training Acc: {training_accuracy:.4f}, Validation Acc: {val_accuracy:.4f}")


    def _compute_loss(self, X, y):
        linear_model = np.dot(X, self.weights) + self.bias
        y_predicted = self.softmax(linear_model)
        return -np.mean(np.log(y_predicted[range(len(y)), y]))  

    def _compute_accuracy(self, X, y):
        y_predicted = self.predict(X)
        return np.mean(y_predicted == y)

    def predict(self, X):
        linear_model = np.dot(X, self.weights) + self.bias
        y_predicted = np.argmax(self.softmax(linear_model), axis=1)
        return y_predicted
