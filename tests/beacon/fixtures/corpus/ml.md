# Machine Learning Fundamentals

Machine learning is a subfield of artificial intelligence in which algorithms learn patterns from data instead of being explicitly programmed.
Supervised learning trains a model on labeled examples to predict outputs for unseen inputs.
Unsupervised learning discovers structure in unlabeled data, such as clustering similar records together.

Neural networks are composed of layers of interconnected nodes called neurons.
Each neuron applies a linear transformation followed by a non-linear activation function.
Deep learning refers to neural networks with many hidden layers, capable of learning hierarchical representations.

Gradient descent is the core optimization algorithm used to train neural networks.
It iteratively adjusts model parameters in the direction that reduces the loss function.
The learning rate controls how large each step is during gradient descent.
Too large a learning rate causes divergence; too small causes slow convergence.

Training a model requires splitting data into train, validation, and test sets.
Overfitting occurs when a model memorizes training data but generalizes poorly.
Regularization techniques like dropout and weight decay reduce overfitting.
Batch normalization stabilizes training by normalizing activations within each mini-batch.
