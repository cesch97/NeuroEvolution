# Neuro Evolution in Julia

Training a neural network to classify images of digits from the MNIST dataset without the use of backpropagation but with an evolution strategy. The strategy used in this example is a simplified version of the one developed by OpenAi and publicated on [their blog](https://openai.com/blog/evolution-strategies/) and on [arXiv](https://arxiv.org/pdf/1703.03864.pdf)
  
### Prerequisites

To run this project you need: [Julia](https://julialang.org/) v1.0 or higher and the following Packages:
- MLDatasets
- Plots
- Printf

## Getting Started

The most common approaches to train a neural network make use of  backpropagation to track the gradient. Another way is to employ evolution strategies, when it comes to supervised learning evolution approaches are inefficient compared to backpropagation but their freedom from derivability make them suitable for complex scenarios when it is impossible to track gradient (like Reinforcement learning).  
Anyway in order to update the parameters of our network we need something similar to the gradient that tells us the direction and the magnitude of the changes that we should apply to each parameter to get a better model. Instead of the real gradient (which can be computed only through backpropagation) we will compute an approximation and then we use it to update our model's parameters through Stochastic Gradient Descent (SGD).  
  
### Intuition  
1. At first we create one model initialized with random parameters sampled from a normal distribution of *mean* and *std* known.
2. Then applying **random mutations** to the model we make a **population**, a set of models slighty different from each others. 
3. We measure how each model performed over the same batch of data and we store this information into the **fitness** vector, we normalize the vector by subtracting it's mean and dividing for its standard deviation to get the **advantage** vector
4. Then we approximate the **gradient** as a weighted sum between each random noise applied and the advantage of moving in that direction.
5. The model's paramaters are then updated by Stochastic Gradient Descent. 
  
### Notes
- We need to keep track of the random noise used for each model of the population in order to approximate gradient.
- The random noise can be thought as a movement in a certain direction in the complex paramaters space of the model. 
- By keeping track of all the directions that we tried in the population (keeping track of the random noise) and knowing the advantage of every movement we took we can compute an approximation of the gradient
  
### Implementation
- I could have used the packages Statistics, Flux, Lathe and avoid writing all the functions myself in the "utils.jl" file but I wanted to show how powerful is Julia and how easy is to implement everything by yourself.

### Results
In the **results** directory there are the output of a 100 epoch training that took about 40 mins on my laptop and reached an accurcy of 89% on the test set. I also put into the directory "examples" some random images from the test set and the model predictions for each of them.

## Authors

* **Fabio Cescon** - [GitHub](https://github.com/cesch97)


## Acknowledgments

* A wonderful [project](https://github.com/d9w/evolution) made by Dennis G. Wilson. It's been my starting point into the world of evolutional computing.
* [Julia](https://julialang.org/)
* [OpenAI](https://openai.com/)

