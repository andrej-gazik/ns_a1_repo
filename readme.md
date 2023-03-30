# Project 1 documentation
[Report preview link](https://api.wandb.ai/links/neural_networks_fiit/a3dr9a3t)
# Design of Neural Network

# Introduction

The goal of this project is to create a neural network that can accurately predict the price range of a mobile phone based on its technical specifications. To reach this goal, we will look at different neural network design techniques and algorithms, such as deep learning architectures, stochastic gradient descent (SGD), and adaptive learning rate methods (Adam).

In this documentation, we will provide an overview of the project and its objectives, as well as a detailed description of the dataset used to train and evaluate the neural network. We will also talk about how the neural network was made, including how the right architecture, hyperparameters, and optimization methods were chosen.

# Problem Statement

The problem at hand is to classify the price range of mobile phones based on their technical specifications. The dataset contains 21 columns of data that describe various aspects of the mobile phone, such as its battery capacity, camera quality, internal storage, RAM, screen size, and other features.

The challenge is to design a neural network that can effectively learn the patterns in the data and make accurate predictions of the price range of the mobile phone. The neural network must be able to handle the complexity and diversity of the dataset while avoiding overfitting and achieving high accuracy.

# Preprocessing

### Data Cleaning:

No extra data cleaning is to be done from our side, as no data is missing.

Our dataset has shape (2000, 21)

### Feature Scaling

Features in our dataset were not normally distrusted; this fact is supported by testing with the Shapiro test; therefore, we have used a standard scaler for scaling the dataset.

### Feature Encoding

Only categorical data is in our output, which consists of values from 0 to 3, categorizing mobile phone prices as follows:

- 0 (low cost)
- 1 (medium cost)
- 2 (high cost)
- 3 (very high cost)

To encode features, we have selected OneHotEncoder, which converts categorical data into numerical form by creating binary columns for each category, making categorical data compatible with our output from the neural network.

### Feature Selection

We have explored if our dataset does contain some features which are more significant than others

Correlation matrix and calculated correlations of all inputs with output have ordered by

### Data Splitting

To test how well the neural network works, the dataset should be split into training, validation, and testing sets. The training set is used to train the neural network, the validation set is used to tune the hyperparameters of the neural network, and the testing set is used to evaluate the performance of the neural network on unseen data.

| test | 70 |
| --- | --- |
| validation | 15 |
| test | 15 |

# Neural Network Architecture

## Input layer

Our input layer consists of 20 neurons representing features in provided datasets. No columns were removed from EDA, therefore the number of input neurons was constant throughout the whole time input_shape = 20..

## Hidden layers

In our solutions, we have explored multiple shapes of hidden layers, as follows:

| Shapes |  |
| --- | --- |
| [] | No hidden |
| [10] | Very short |
| [16,8] | Short  |
| [10,5] | Short  |
| [17, 14, 11, 8] | Medium |
| [18, 12, 8, 6] | Medium |
| [18,16,14,12,10,8,6] | Long |
| [19,18,17,16,15,14,13,12,11,10,9,8,7,6,5] | Very long |

Models with the shapes of hidden layers were built and tested with different parameters.

## Output layer

We chose an approach that uses classification with a softmax activation function on four output neurons to categorize 20 input features into four different price categories.

We used a softmax activation function on the neural network's output layer to put the range of mobile phone prices into one of four groups: low cost, medium cost, high cost, and very high cost. This method gives us four different probabilities as a result. The highest probability shows the predicted class for the price range of the mobile phone we gave as an input.

# Evaluation Metrics

We have used the CrossEntropyLoss function as our loss function for the neural network because we are dealing with categorical data. The CrossEntropyLoss function is often used to sort inputs into different groups. It's a good choice for our problem because it not only measures the difference between predicted and actual values but also takes into account the probabilities assigned to each class. This is important because our neural network is outputting probabilities for each price range category, and we want to make sure that it's not just making a binary decision but rather assigning probabilities to each category.

# Hyperparameter Tuning

For hyperparameter tuning, we have used wandb sweep. Wandb sweep allowed us to easily search through different hyperparameters and architectures by automatically running multiple experiments with different combinations of hyperparameters. This allowed us to quickly find the best hyperparameters and architecture for our neural network.

# Results

Results can be found in [wandb.ai](http://wandb.ai) dashboard or generated report in `/docs` folder.

# Conclusion

Using different neural network design techniques and algorithms, we were able to build a neural network that can accurately predict the price range of mobile phones based on their technical specs.
