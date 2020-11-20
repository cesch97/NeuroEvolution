################################ 
#                              #
#   Neuro-Evolution in Julia   #
#                              #
################################


using MLDatasets
using Plots
using Printf
include("utils.jl")


### Hyper-Parameters ###
batch_size    = 64
hiden_layers  = [16, 16]
pop_size      = 8
num_epochs    = 100
σ_init_params = 0.1
σ_exp_noise   = 0.001
lr            = 0.001
results_dir   = "./results"
# - - - - - - - - - - -#


### Data ###
x_train, y_train = MNIST.traindata()
x_test, y_test = MNIST.testdata()

x_train = Float32.(x_train)
x_test = Float32.(x_test)

x_train = reshape(x_train, (28*28, :))'
x_test = reshape(x_test, (28*28, :))'

y_train = onehot(y_train)
y_test = onehot(y_test)

train_batches = make_batches(x_train, y_train, batch_size)


### Model ###
model = init_model(784, 10, hiden_layers, σ_init_params)


### Evolution ###
loss_hist, acc_hist = evolve!(model, train_batches, pop_size, num_epochs, σ_exp_noise, lr)


### Results ###
plot_metrics(loss_hist, acc_hist, results_dir)
idx = rand(16:size(x_test, 1))
plot_imgs(x_train[idx-15:idx, :], onecold(forward(model, x_train[idx-15:idx, :])), results_dir)

train_loss = mse_loss(y_train, forward(model, x_train))
test_loss = mse_loss(y_test, forward(model, x_test))
train_acc = evaluate(model, (x=x_train, y=y_train))
test_acc = evaluate(model, (x=x_test, y=y_test))
println("Results:")
println("- train data:")
@printf("   - loss -> %.3f\n", train_loss)
@printf("   - acc  -> %.3f\n", train_acc)
println("- test data:")
@printf("   - loss -> %.3f\n", test_loss)
@printf("   - acc  -> %.3f\n", test_acc)