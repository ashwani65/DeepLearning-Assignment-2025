import numpy as np
from tensorflow.keras.datasets import mnist 
import matplotlib.pyplot as plt

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

def binarize(image, threshold=127):
    # Convert image (which is a NumPy array) to binary using a comparison and cast to float type.
    binary_image = (image >= threshold).astype(np.float32)
    return binary_image

train_images_bin = np.array([binarize(img) for img in train_images])
test_images_bin = np.array([binarize(img) for img in test_images])

target_class = 6
# Filter training images: select only those with label equal to target_class (8)
train_filter = (train_labels == target_class)
train_images_filtered = train_images_bin[train_filter]
train_labels_filtered = train_labels[train_filter]

# Filter testing images similarly
test_filter = (test_labels == target_class)
test_images_filtered = test_images_bin[test_filter]
test_labels_filtered = test_labels[test_filter]

num_train = 5500
if len(train_images_filtered) < num_train:
    raise ValueError("Not enough images for training after filtering by class 8.")


# Use the first 5500 images for training; the rest can be optionally added to test set.
train_data = train_images_filtered[:num_train]
# For test data, we can combine the leftover training images (if any) with the filtered test set.
test_data = np.concatenate([train_images_filtered[num_train:], test_images_filtered], axis=0)

print("Data Preparation Complete:")
print("Training images (class 8):", train_data.shape)
print("Testing images (class 8):", test_data.shape)

class RBM:
    def __init__(self, n_visible=784, n_hidden=64, learning_rate=0.1):
        self.n_visible = n_visible
        self.n_hidden = n_hidden
        self.learning_rate = learning_rate
        self.W = np.random.normal(0, 0.01, (n_visible, n_hidden))  # Weight matrix
        self.b = np.zeros(n_visible)  # Visible biases
        self.c = np.zeros(n_hidden)   # Hidden biases

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    
    def sample_hidden(self, v):
        # Given a visible vector v (flattened), compute hidden probabilities and sample a binary hidden vector.
        activations = np.dot(v, self.W) + self.c
        prob_h = self.sigmoid(activations)
        h_sample = (prob_h >= np.random.rand(self.n_hidden)).astype(np.float32)
        return prob_h, h_sample

    def sample_visible(self, h):
        # Given a hidden vector h, compute visible probabilities and sample a binary visible vector.
        activations = np.dot(h, self.W.T) + self.b
        prob_v = self.sigmoid(activations)
        v_sample = (prob_v >= np.random.rand(self.n_visible)).astype(np.float32)
        return prob_v, v_sample

    def free_energy(self, v):
        # Compute the free energy for a visible vector v.
        wx_b = np.dot(v, self.W) + self.c  # shape: (n_hidden,)
        # Free energy: F(v) = - b^T v - sum_j log(1 + exp(wx_b_j))
        return - np.dot(v, self.b) - np.sum(np.log(1 + np.exp(wx_b)))

    def pseudo_likelihood(self, data):
        # Compute an approximate pseudo log-likelihood over the dataset.
        # For each sample, flip a random visible unit and compute the free energy difference.
        n_visible = self.n_visible
        total_pl = 0.0
        for v in data:
            v = v.flatten()
            # Select a random index to flip
            i = np.random.randint(n_visible)
            v_flip = np.copy(v)
            v_flip[i] = 1 - v_flip[i]  # Flip bit
            fe_v = self.free_energy(v)
            fe_v_flip = self.free_energy(v_flip)
            pl = n_visible * np.log(self.sigmoid(fe_v_flip - fe_v))
            total_pl += pl
        return total_pl / len(data)

    def contrastive_divergence(self, v_input):
        # Perform one step of Contrastive Divergence (CD-1) for one training sample.
        # Positive phase: compute hidden activations and sample h
        pos_hidden_prob, pos_hidden_sample = self.sample_hidden(v_input)
        pos_grad = np.outer(v_input, pos_hidden_prob)

        # Negative phase: reconstruct visible units and recompute hidden activations
        neg_visible_prob, neg_visible_sample = self.sample_visible(pos_hidden_sample)
        neg_hidden_prob, _ = self.sample_hidden(neg_visible_sample)
        neg_grad = np.outer(neg_visible_sample, neg_hidden_prob)

        # Update weights and biases
        self.W += self.learning_rate * (pos_grad - neg_grad)
        self.b += self.learning_rate * (v_input - neg_visible_sample)
        self.c += self.learning_rate * (pos_hidden_prob - neg_hidden_prob)

        # Use squared error of reconstruction as a simple error measure
        error = np.sum((v_input - neg_visible_prob) ** 2)
        return error

    def train(self, data, epochs=50):
        # Train the RBM on the provided data (each sample should be flattened to a vector).
        errors = []
        pseudo_likelihoods = []
        num_samples = data.shape[0]
        for epoch in range(epochs):
            epoch_error = 0.0
            for i in range(num_samples):
                v_input = data[i].flatten()
                error = self.contrastive_divergence(v_input)
                epoch_error += error
            avg_error = epoch_error / num_samples
            errors.append(avg_error)
            
            # Compute pseudo log-likelihood as an approximation of log-likelihood
            pl = self.pseudo_likelihood(data)
            pseudo_likelihoods.append(pl)
            print(f"Epoch {epoch+1}/{epochs}, Reconstruction Error: {avg_error:.4f}, Pseudo Log-Likelihood: {pl:.4f}")
        return errors, pseudo_likelihoods

# Create an RBM instance with 784 visible units (28x28) and 64 hidden units.
rbm = RBM(n_visible=784, n_hidden=64, learning_rate=0.1)

# (For demonstration, here is how you would start training using the prepared train_data.)
# We assume train_data is an array of shape (5500, 28, 28) and will be flattened inside the training.
epochs = 50  # For now, we use 10 epochs as an example.
training_errors, pseudo_lls = rbm.train(train_data, epochs=epochs)
print("RBM Training (Steps 1-3) complete.")

plt.figure()
plt.plot(range(1, epochs+1), pseudo_lls, marker='o')
plt.xlabel('Epoch')
plt.ylabel('Pseudo Log-Likelihood')
plt.title('Pseudo Log-Likelihood vs. Epoch')
plt.show()

# Hidden Space Interpolation Experiment
def hidden_interpolation(rbm, num_pairs=20):
    alphas = [0, 0.2, 0.4, 0.6, 0.8, 1]
    for pair in range(num_pairs):
        # Randomly sample two binary hidden vectors h1 and h2
        h1 = np.random.randint(0, 2, size=rbm.n_hidden).astype(np.float32)
        h2 = np.random.randint(0, 2, size=rbm.n_hidden).astype(np.float32)
        images = []
        for alpha in alphas:
            # Interpolate: h_alpha = alpha * h1 + (1-alpha)*h2, then threshold at 0.5
            h_alpha = alpha * h1 + (1 - alpha) * h2
            h_alpha_bin = (h_alpha >= 0.5).astype(np.float32)
            # Sample visible vector from h_alpha_bin
            _, v_sample = rbm.sample_visible(h_alpha_bin)
            images.append(v_sample.reshape(28, 28))
        
        # Plot the 6 images (one for each alpha) in a row
        plt.figure(figsize=(12, 2))
        for i, img in enumerate(images):
            plt.subplot(1, len(images), i + 1)
            plt.imshow(img, cmap='gray')
            plt.title(f'alpha={alphas[i]}')
            plt.axis('off')
        plt.suptitle(f'Interpolation Pair {pair+1}')
        plt.show()

print("Starting Hidden Space Interpolation Experiment...")
hidden_interpolation(rbm, num_pairs=20)


# Hidden Perturbation Experiment on Test Images
def hidden_perturbation(rbm, test_data, num_samples=20):
    indices = np.random.choice(len(test_data), size=num_samples, replace=False)
    for idx in indices:
        v_x = test_data[idx].flatten()
        # Compute hidden representation for v_x
        _, h_x = rbm.sample_hidden(v_x)
        # Create a random binary vector h_delta with exactly 10 ones
        h_delta = np.zeros(rbm.n_hidden)
        ones_indices = np.random.choice(rbm.n_hidden, size=10, replace=False)
        h_delta[ones_indices] = 1
        # Perturb hidden state: add h_delta and clip to ensure binary values
        h_new = np.clip(h_x + h_delta, 0, 1)
        # Sample a new visible vector from the perturbed hidden state
        _, v_delta = rbm.sample_visible(h_new)
        # Compute L2 distance between original and perturbed visible vectors
        l2_distance = np.linalg.norm(v_x - v_delta)
        
        # Plot the original and perturbed images side by side
        plt.figure(figsize=(6, 3))
        plt.subplot(1, 2, 1)
        plt.imshow(v_x.reshape(28, 28), cmap='gray')
        plt.title('Original v_x')
        plt.axis('off')
        plt.subplot(1, 2, 2)
        plt.imshow(v_delta.reshape(28, 28), cmap='gray')
        plt.title(f'Perturbed v_δ\nL2 = {l2_distance:.2f}')
        plt.axis('off')
        plt.suptitle(f'Test Sample Index: {idx}')
        plt.show()

print("Starting Hidden Perturbation Experiment on Test Images...")
hidden_perturbation(rbm, test_data, num_samples=20)
