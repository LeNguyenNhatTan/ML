import numpy as np


class SoftmaxRegression:
    def __init__(self, learning_rate=0.01, n_iter=1000, batch_size=32):
        self.learning_rate = learning_rate
        self.n_iter = n_iter
        self.batch_size = batch_size
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
    
    def _compute_loss(self, X, y, n_classes):
        linear_model = np.dot(X, self.weights) + self.bias
        y_predicted = self.softmax(linear_model)
        y_encoded = self.one_hot_encode(y, n_classes)
        return -np.mean(np.sum(y_encoded * np.log(y_predicted), axis=1))

    def _compute_accuracy(self, X, y):
        y_predicted = self.predict(X)
        return np.mean(y_predicted == y)

    def predict(self, X):
        linear_model = np.dot(X, self.weights) + self.bias
        y_predicted = np.argmax(self.softmax(linear_model), axis=1)
        return y_predicted
    
    def predict_proba(self, X):
        linear_model = np.dot(X, self.weights) + self.bias
        return self.softmax(linear_model)
    
    def fit(self, X, y, X_val, y_val):
        n_samples, n_features = X.shape
        n_classes = len(np.unique(y))

        self.weights = np.zeros((n_features, n_classes))
        self.bias = np.zeros(n_classes)
        
        for i in range(self.n_iter):
            # Shuffle the data to ensure randomization for each mini-batch
            indices = np.random.permutation(n_samples)
            X_shuffled = X[indices]
            y_shuffled = y[indices]
            
            epoch_training_loss = []
            epoch_training_accuracy = []

            # Loop over mini-batches
            for start_idx in range(0, n_samples, self.batch_size):
                end_idx = min(start_idx + self.batch_size, n_samples)
                X_batch = X_shuffled[start_idx:end_idx]
                y_batch = y_shuffled[start_idx:end_idx]

                # Compute predictions for the mini-batch
                linear_model = np.dot(X_batch, self.weights) + self.bias
                y_predicted = self.softmax(linear_model)


                # Update weights and bias
                dw = (1 / len(y_batch)) * np.dot(X_batch.T, (y_predicted - self.one_hot_encode(y_batch, n_classes)))
                db = (1 / len(y_batch)) * np.sum(y_predicted - self.one_hot_encode(y_batch, n_classes), axis=0)
                self.weights -= self.learning_rate * dw
                self.bias -= self.learning_rate * db
               
                # Compute and store mini-batch loss and accuracy
                batch_loss = self._compute_loss(X_batch, y_batch, n_classes)
                batch_accuracy = self._compute_accuracy(X_batch, y_batch)
                epoch_training_loss.append(batch_loss)
                epoch_training_accuracy.append(batch_accuracy)

            # Record average loss and accuracy for the epoch
            avg_training_loss = np.mean(epoch_training_loss)
            avg_training_accuracy = np.mean(epoch_training_accuracy)
            self.training_losses.append(avg_training_loss)
            self.training_accuracies.append(avg_training_accuracy)

            # Compute loss and accuracy on validation set
            val_loss = self._compute_loss(X_val, y_val, n_classes)
            val_accuracy = self._compute_accuracy(X_val, y_val)

            self.validation_losses.append(val_loss)
            self.validation_accuracies.append(val_accuracy)

            if i % 100 == 0:
                print(f"Epoch {i}, Training Loss: {avg_training_loss:.4f}, Validation Loss: {val_loss:.4f}, "
                      f"Training Acc: {avg_training_accuracy:.4f}, Validation Acc: {val_accuracy:.4f}")

    
    # # Waiting
    # def create_balanced_batches(self, X, y):
    #     # X: dữ liệu đầu vào (features), dạng numpy array
    #     # y: nhãn tương ứng (labels), dạng numpy array
    #     # batch_size: kích thước mỗi batch

    #     # Bước 1: Phân nhóm dữ liệu theo lớp
    #     classes = np.unique(y)
    #     class_indices = {cls: np.where(y == cls)[0] for cls in classes}

    #     # Bước 2: Random hóa thứ tự trong từng lớp
    #     for cls in classes:
    #         np.random.shuffle(class_indices[cls])

    #     # Bước 3: Chia batch
    #     batches = []
    #     while True:
    #         batch_indices = []
    #         for cls in classes:
    #             # Lấy số mẫu cần từ từng lớp để đạt được cân bằng
    #             take_count = min(len(class_indices[cls]), self.batch_size // len(classes))
    #             if take_count > 0:
    #                 batch_indices.extend(class_indices[cls][:take_count])
    #                 class_indices[cls] = class_indices[cls][take_count:]
            
    #         if len(batch_indices) == 0:
    #             break  # Hết mẫu
            
    #         np.random.shuffle(batch_indices)  # Random hóa thứ tự trong batch
    #         batches.append((X[batch_indices], y[batch_indices]))

    #     return batches