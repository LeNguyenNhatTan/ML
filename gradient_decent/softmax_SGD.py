import numpy as np

class SoftmaxRegression:
    def __init__(self, learning_rate=0.01, n_iter=1000, patience=10):
        self.learning_rate = learning_rate
        self.n_iter = n_iter
        self.patience = patience  # Số epoch không cải thiện trước khi dừng
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

        # Biến đếm cho early stopping
        count = 0
        best_loss = float('inf')

        for i in range(self.n_iter):
            for j in range(n_samples):
                Xi = X[j:j+1]  
                yi = y[j:j+1]  

                # Tính toán dự đoán cho mẫu hiện tại
                linear_model = np.dot(Xi, self.weights) + self.bias
                y_predicted = self.softmax(linear_model)

                # Tính toán độ dời và trọng số cho mẫu đơn
                dw = np.dot(Xi.T, (y_predicted - self.one_hot_encode(yi, n_classes)))
                db = y_predicted - self.one_hot_encode(yi, n_classes)

                # Cập nhật trọng số và độ dời
                self.weights -= self.learning_rate * dw
                self.bias -= self.learning_rate * db.squeeze()

            # Tính toán loss cho tập train sau mỗi epoch
            training_loss = self._compute_loss(X, y)
            self.training_losses.append(training_loss)

            if training_loss < best_loss:
                best_loss = training_loss
                count = 0  
            else:
                count += 1

            if count >= self.patience:
                print(f"Early stopping at epoch {i} with training loss: {training_loss:.4f}")
                break

            # Tính toán loss và accuracy trên tập validation
            val_loss = self._compute_loss(X_val, y_val)
            val_accuracy = self._compute_accuracy(X_val, y_val)
            self.validation_losses.append(val_loss)
            self.validation_accuracies.append(val_accuracy)

            # In thông tin mỗi 100 epochs
            if i % 100 == 0:
                training_accuracy = self._compute_accuracy(X, y)
                print(f"Epoch {i}, Training Loss: {training_loss:.4f}, Validation Loss: {val_loss:.4f}, "
                      f"Training Acc: {training_accuracy:.4f}, Validation Acc: {val_accuracy:.4f}")

    def _compute_loss(self, X, y):
        linear_model = np.dot(X, self.weights) + self.bias
        y_predicted = self.softmax(linear_model)
        y_encoded = self.one_hot_encode(y, len(np.unique(y)))
        return -np.mean(np.sum(y_encoded * np.log(y_predicted), axis=1))

    def _compute_accuracy(self, X, y):
        y_predicted = self.predict(X)
        return np.mean(y_predicted == y)

    def predict(self, X):
        linear_model = np.dot(X, self.weights) + self.bias
        y_predicted = np.argmax(self.softmax(linear_model), axis=1)
        return y_predicted
