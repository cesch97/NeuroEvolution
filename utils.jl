

### Statistics ###

function mean(x::Vector)
    sum(x) / length(x)
end

function variance(x::Vector)
    x_mean = mean(x)
    sum(broadcast(x -> (x - x_mean)^2, x)) / length(x)
end

function std(x::Vector)
    sqrt(variance(x))
end



### Data preprocessing ###

function onehot(y::Vector{Int})
    #=
    Apply one-hot encoding to a vector of integers
    =#
    n_features = maximum(y) + 1
    y_onehot = zeros(length(y), n_features)
    for i in eachindex(y)
        y_onehot[i, y[i] + 1] = 1.
    end
    Float32.(y_onehot)
end

function onecold(y_onehot::AbstractMatrix)
    #=
    decode one-hot matrix to an integer vector
    =#
    n_features = size(y_onehot, 2)
    y = Int[]
    for i in 1:size(y_onehot, 1)
        push!(y, argmax(y_onehot[i, :]) - 1)
    end
    y
end

function make_batches(x::AbstractMatrix, y::AbstractMatrix, batch_size)
    #=
    divide the dataset into a list of batches, the last batch
    could not have size == batch_size
    =#
    batches = NamedTuple[]
    i = batch_size
    while i < size(x, 1)
        batch = (x = x[i - batch_size + 1: i, :], 
                 y = y[i - batch_size + 1: i, :])
        push!(batches, batch)
        i += batch_size
    end
    if i != size(x, 1)
        batch = (x = x[i - batch_size + 1: end, :], 
                 y = y[i - batch_size + 1: end, :])
        push!(batches, batch) 
    end
    batches
end



### Neural Networks ###

mutable struct LinearLayer
    #=
    This is an object that represent a 
    fully connected layer of a NN, it stores
    the layer parameters
    =#
    w::AbstractMatrix
    b::Array{Float32,1}
end

function init_model(input_dim::Int, out_dim::Int, h_layers::Array{Int}, σ::Float64)
    #=
    Returns a list of LinearLayers with paramaters
    initiaded from a normal distribution with μ = 0 and std = σ  
    =#
    model = LinearLayer[]
    push!(model, LinearLayer(Float32.(randn(h_layers[1], input_dim) .* σ),
                             Float32.(randn(h_layers[1]) .* σ)))
    for i in 1:length(h_layers) - 1
        push!(model, LinearLayer(Float32.(randn(h_layers[i + 1], h_layers[i]) .* σ),
                                 Float32.(randn(h_layers[i + 1]) .* σ)))
    end
    push!(model, LinearLayer(Float32.(randn(out_dim, h_layers[end]) .* σ),
                             Float32.(randn(out_dim) .* σ)))
    model
end

function relu(z::Float32)
    #=
    The relu activation function
    return z if z > 0 else 0
    =#
    z > 0 ? z : 0
end

function forward(model::Array{LinearLayer}, x::AbstractMatrix)
    #=
    Run a forward pass of the input x throw the model
    the activation function is not applied to the output layer
    =#
    x = x'
    for layer in model[1: end-1]
        x = relu.(layer.w * x .+ layer.b)
    end
    x = model[end].w * x .+ model[end].b 
    x'
end



### Metrics ###

function mse_loss(y_pred::AbstractMatrix, y::AbstractMatrix)
    #=
    Compute the Mean Square Error loss over two matricies
    (one should be the output of the network over a batch of data,
    the other the one-hot version of the labels)
    =#
    mean(dropdims(sum((y - y_pred).^2, dims=2), dims=2))
end

function accuracy(y_pred::Array{Int,1}, y::Array{Int,1})
    #=
    Compute the ratio between the right predictions
    and all the predictions
    =#
    accuracy = 0
    for (i, j) in zip(y_pred, y)
        (i == j) && (accuracy += 1)
    end
    accuracy / length(y)
end

function evaluate(model::Array{LinearLayer,1}, batch::NamedTuple)
    #=
    Return the accuracy of the model over a batch of data
    =#
    out = forward(model, batch.x)
    out = onecold(out)
    accuracy(out, onecold(batch.y))
end



### Evolution strategy ###

function get_exp_noise(model::Array{LinearLayer}, σ::Float64)
    #=
    Given a model it returns a list of NemdTuples,
    each tutple contains a 'w' and 'b' parameter 
    with the same size of their LinearLayer counterpart.
    They store the exploration noise that will be added
    to the original model to apply mutation
    =#
    exp_noise = NamedTuple[]
    for layer in model
        w_noise = Float32.(randn(size(layer.w)) .* σ)
        b_noise = Float32.(randn(size(layer.b)) .* σ)
        push!(exp_noise, (w = w_noise, b = b_noise))
    end
    exp_noise
end

function mutate(model::Array{LinearLayer,1}, exp_noise::Array{NamedTuple,1})
    #=
    It adds the exp_noise to each layer of the model
    and returns a new muteted model 
    =#
    mut_model = LinearLayer[]
    for i in eachindex(model)
        push!(mut_model, LinearLayer(model[i].w + exp_noise[i].w,
                                     model[i].b + exp_noise[i].b))
    end
    mut_model
end

function compute_grad_approx(exp_noise::Array{Array{NamedTuple,1},1}, advantage::Array{Float32,1})
    #=
    Return the approximated gradient for each layer of the model,
    the gradient is approximated as the weighted sum between
    the exp_noise and its relative advantage
    =#
    grad = NamedTuple[]
    for i in 1:length(exp_noise[1])
        w_grad = sum([exp_noise[j][i].w .* advantage[j] for j in 1:length(exp_noise)])
        b_grad = sum([exp_noise[j][i].b .* advantage[j] for j in 1:length(exp_noise)])
        push!(grad, (w = w_grad, b = b_grad))
    end
    grad
end

function sgd_update!(model::Array{LinearLayer,1}, grad::Array{NamedTuple,1}, 
                     lr::Float64, σ::Float64, pop_size::Int)
    #=
    Using the approximated gradient apply one step of 
    Sstochastic Gradient Descent over the model parameters
    =#
    for i in 1:length(model)
        model[i].w -= grad[i].w .* Float32.(lr / (pop_size * σ))
        model[i].b -= grad[i].b .* Float32.(lr / (pop_size * σ))
    end
end

function evolve!(model::Array{LinearLayer,1}, batches::Array{NamedTuple,1},
                pop_size::Int, num_epochs::Int, σ::Float64, lr::Float64)
    #=
    Evolving the model for a certain number of epochs
    =#
    loss_hist = Float64[]
    acc_hist = Float64[]
    for epoch in 1:num_epochs
        losses = Float64[]
        accuracies = Float64[]
        for i in eachindex(batches)
            batch = batches[i]
            # create a population by applying exp_noise
            # to copies of the model
            population = [model for i in 1:pop_size]
            exp_noise = get_exp_noise.(population, [σ])
            population = mutate.(population, exp_noise)
            # feed a batch of data to every model in the population
            # and compute the MSE loss for each one
            outs = forward.(population, [batch.x])
            fitness = mse_loss.(outs, [batch.y])
            # the advantage is just the normalization of the fitness
            advantage = Float32.((fitness .- mean(fitness)) ./ std(fitness))
            grad = compute_grad_approx(exp_noise, advantage)
            sgd_update!(model, grad, lr, σ, pop_size)
            push!(losses, mse_loss(forward(model, batch.x), batch.y))
            push!(accuracies, evaluate(model, batch))
        end
        loss_mean = mean(losses)
        acc_mean = mean(accuracies)
        @printf("Epoch: %i, loss: %.3f, accuracy: %.3f\n", epoch, loss_mean, acc_mean)
        push!(loss_hist, loss_mean)
        push!(acc_hist, acc_mean)
    end 
    println("")
    loss_hist, acc_hist
end



### Plotting ###

function plot_metrics(loss::Array{Float64,1}, acc::Array{Float64,1},
                      results_dir::String="")
    #=
    Plot a chart with the loss and the accuracy of the model
    at each epoch of the evolution process and save it as 
    a .png file
    =#
    p_loss = plot(loss, label="loss")
    p_acc = plot(acc .* 100, label="accuracy", legend=:bottomright)
    p = plot(p_loss, p_acc, layout=(2, 1))
    if results_dir != ""
        savefig(p, results_dir*"/metrics.png")
    end
    p
end

function plot_imgs(x::AbstractMatrix, y_pred::Array{Int,1},
                   results_dir::String="")
    #=
    Plot a grid of 4x4 images from the dataset with the predictions
    of the model for the image and save it as .png file
    =#
    @assert size(x, 1) == 16
    plots = []
    for i in 1:16
        p = plot(Gray.(reshape(x[i, :], (28, 28))'), ticks=false)
        title!(p, "$(y_pred[i])")
        push!(plots, p)
    end
    p = plot(plots..., layout=(4, 4), legend=false)
    if results_dir != ""
        savefig(p, results_dir*"/imges.png")
    end
    p
end