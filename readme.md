# Neuro Evolution in Julia

Training a neural network to classify digit images of the MNIST dataset without backpropagation but with an evolution strategy. The strategy used is a simplified version of the one developed by OpenAi and publicated on [their blog](https://openai.com/blog/evolution-strategies/) and on [arXiv](https://arxiv.org/pdf/1703.03864.pdf)
  
### Prerequisites

To run this project you need: [Julia](https://julialang.org/) v1.0 or higher and the following Packages:
- MLDatasets
- Plots
- Printf

## Getting Started

The most common approach to train a neural network is with backpropagation. Another way is to use evolution strategies, this approach is inefficient compared to backpropagation when it comes to supervised learning but its independency from the gradient makes it suitable for complex scenarios when it's impossible to track gradient (like Reinforcement learning).  
In order to update the parameters of our network we need something like the gradient that tells us the direction and the magnitude of the update that we should take for each parameter. Instead of the real gradient (which can be computed only throw backpropagation) we will use an approximation of the gradient and then we use that approximation to update our model's parameters.  
  
### Intuition  
1. At first we create one model initialized with random numbers sampled from a normal distribution of mean and std known.
2. Then we create a **population** by applying a **random mutation** to the model and making a set of models slighty different from each others. 
3. We measure how each model performed over a batch of data and we store this information into a vector, we normalize the vector by subtracting it's mean and dividing for its standard deviation to get the **advantage** vector
4. Then we approximate the gradient as a weighted sum between each random noise applied and the advantage of moving in that direction.
5. The model's paramaters are then updated by Stochastic Gradient Descent. 

### Note
- We need to keep track of the random noise used for each model of the population in order to approximate gradient.
- The random noise can be think as a movement in a certain direction in the complex paramater space of the model. 
- By keeping track of all the directions that we tried in the population (keeping track of the random noise) and knowing the advantage of every movement we can compute an approimation of the gradient

## Authors

* **Fabio Cescon** - [GitHub](https://github.com/cesch97)


## Acknowledgments

* A wonderful [project](https://github.com/d9w/evolution) made by Dennis G. Wilson. It's been my starting point into the world of evolutional computing.
* [OpenAI](https://openai.com/)

