#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "neuralnet.h"

// Returns a random float from a normal distribution with the given mean and standard deviation,
// using the Box-Muller transform. Used for mutation noise, since small nudges should be common
// and large jumps rare.
float random_gaussian(float mean, float stddev) {
    float u1 = (float)rand() / (float)RAND_MAX;
    float u2 = (float)rand() / (float)RAND_MAX;
    if (u1 < 1e-7f) u1 = 1e-7f; // Avoid log(0)

    float z = sqrtf(-2.0f * logf(u1)) * cosf(2.0f * (float)M_PI * u2);
    return mean + z * stddev;
}

// Returns a random float uniformly distributed in [-limit, limit].
float random_uniform(float limit) {
    float r = (float)rand() / (float)RAND_MAX; // 0.0 to 1.0
    return (r * 2.0f - 1.0f) * limit; // -limit to limit
}

// Function for creating a new layer.
// Takes in an input size, output size, and activation type.
// Weights are randomly initialized to break symmetry between neurons;
// the initialization range is chosen based on the layer's activation
// (He init for ReLU, Xavier init otherwise). Biases start at zero, which is fine.
Layer create_layer(int input_size, int output_size, Activation activation) {
    Layer layer;
    layer.input_size = input_size;
    layer.output_size = output_size;
    layer.activation = activation;
    layer.weights = malloc(sizeof(float) * (size_t)input_size * output_size);
    layer.biases = calloc((size_t)output_size, sizeof(float));
    layer.outputs = calloc((size_t)output_size, sizeof(float));

    if (!layer.weights || !layer.biases || !layer.outputs) {
        fprintf(stderr, "create_layer: allocation failed\n");
        exit(1);
    }

    float limit;
    if (activation == ACTIVATION_RELU) {
        limit = sqrtf(6.0f / input_size); // He-style
    } else {
        limit = sqrtf(6.0f / (input_size + output_size)); // Xavier-style
    }

    size_t weight_count = (size_t)input_size * output_size;
    for (size_t i = 0; i < weight_count; i++) {
        layer.weights[i] = random_uniform(limit);
    }

    return layer;
}

// Function for creating a new network.
// Takes in the number of layers and a list of layers.
Network create_network(int num_layers, Layer *layers) {
    Network network;
    network.num_layers = num_layers;
    network.layers = malloc(sizeof(Layer) * num_layers);
    memcpy(network.layers, layers, sizeof(Layer) * num_layers);
    return network;
}

// Deep-copies a layer: allocates fresh weights/biases/outputs arrays rather than
// sharing the original's pointers. Needed so mutating a clone never affects the original.
Layer clone_layer(Layer *l) {
    Layer copy;
    copy.input_size = l->input_size;
    copy.output_size = l->output_size;
    copy.activation = l->activation;

    size_t weight_count = (size_t)l->input_size * l->output_size;
    copy.weights = malloc(sizeof(float) * weight_count);
    copy.biases = malloc(sizeof(float) * l->output_size);
    copy.outputs = calloc((size_t)l->output_size, sizeof(float));

    if (!copy.weights || !copy.biases || !copy.outputs) {
        fprintf(stderr, "clone_layer: Allocation failed.\n");
        exit(1);
    }

    memcpy(copy.weights, l->weights, sizeof(float) * weight_count);
    memcpy(copy.biases, l->biases, sizeof(float) * l->output_size);

    return copy;
}

// Deep-copies an entire network, layer by layer.
Network clone_network(Network *net) {
    Network copy;
    copy.num_layers = net->num_layers;
    copy.layers = malloc(sizeof(Layer) * net->num_layers);

    for (int i = 0; i < net->num_layers; i++) {
        copy.layers[i] = clone_layer(&net->layers[i]);
    }

    return copy;
}

// Frees a layer's allocated memory. Does not free the Layer struct itself,
// since Layers are typically stored by value in a Network's layers array.
void free_layer(Layer *l) {
    free(l->weights);
    free(l->biases);
    free(l->outputs);
}

// Frees every layer in a network, then the network's layer array itself.
void free_network(Network *net) {
    for (int i = 0; i < net->num_layers; i++) {
        free_layer(&net->layers[i]);
    }
    free(net->layers);
}

// Randomly perturbs a layer's weights and biases in place.
// Each weight/bias is nudged with probability mutation_rate, by an amount drawn
// from a normal distribution with standard deviation mutation_strength.
void mutate_layer(Layer *l, float mutation_rate, float mutation_strength) {
    size_t weight_count = (size_t)l->input_size * l->output_size;

    for (size_t i = 0; i < weight_count; i++) {
        if ((float)rand() / (float)RAND_MAX < mutation_rate) {
            l->weights[i] += random_gaussian(0.0f, mutation_strength);
        }
    }

    for (int j = 0; j < l->output_size; j++) {
        if ((float)rand() / (float)RAND_MAX < mutation_rate) {
            l->biases[j] += random_gaussian(0.0f, mutation_strength);
        }
    }
}

// Randomly perturbs every layer in a network in place. See mutate_layer.
void mutate_network(Network *net, float mutation_rate, float mutation_strength) {
    for (int i = 0; i < net->num_layers; i++) {
        mutate_layer(&net->layers[i], mutation_rate, mutation_strength);
    }
}

// Function for creating a new training context.
// Stores training information such as the current deltas, weight gradients, and bias gradients.
TrainingContext create_training_context(Network *network) {
    TrainingContext ctx;
    ctx.deltas = malloc(sizeof(float *) * network->num_layers);

    size_t max_weight_count = 0;
    size_t max_output_size = 0;

    for (int l = 0; l < network->num_layers; l++) {
        Layer *layer = &network->layers[l];
        ctx.deltas[l] = malloc(sizeof(float) * layer->output_size);

        size_t weight_count = (size_t)layer->input_size * layer->output_size;
        if (weight_count > max_weight_count) max_weight_count = weight_count;
        if ((size_t)layer->output_size > max_output_size) max_output_size = layer->output_size;
    }

    ctx.weight_grads = malloc(sizeof(float) * max_weight_count);
    ctx.bias_grads = malloc(sizeof(float) * max_output_size);

    return ctx;
}

// Frees allocated memory for a given training context.
void free_training_context(TrainingContext *ctx, int num_layers) {
    for (int l = 0; l < num_layers; l++) {
        free(ctx->deltas[l]);
    }
    free(ctx->deltas);
    free(ctx->weight_grads);
    free(ctx->bias_grads);
}

// Simple ReLU activation function.
// Clamps the value to zero if negative, else returns the input value.
float ReLU(float x) {
    return x > 0 ? x : 0.0f;
}

// The derivative of the ReLU activation function.
float ReLU_derivative(float x) {
    return x > 0 ? 1.0f : 0.0f;
}

// Sigmoid activation function.
float Sigmoid(float x) {
    return 1.0f / (1.0f + expf(-x));
}

// Derivative of Sigmoid, expressed in terms of its own output.
float Sigmoid_derivative(float output) {
    return output * (1.0f - output);
}

// Tanh activation function.
float Tanh(float x) {
    return tanhf(x);
}

// Derivative of tanh, expressed in terms of its own output.
float Tanh_derivative(float output) {
    return 1.0f - (output * output);
}

// Dispatches to the right derivative based on a layer's activation type.
float activation_derivative(Activation activation, float output) {
    switch (activation) {
        case ACTIVATION_RELU:    return ReLU_derivative(output);
        case ACTIVATION_LINEAR:  return 1.0f;
        case ACTIVATION_SIGMOID: return Sigmoid_derivative(output);
        case ACTIVATION_TANH:    return Tanh_derivative(output);
    }
    return 1.0f;  // Unreachable if activation is valid; silences -Wreturn-type
}

// Function for checking if a given activation is valid.
// Used for checking malformed or corrupted saves.
int activation_is_valid(Activation activation) {
    switch (activation) {
        case ACTIVATION_RELU:
        case ACTIVATION_LINEAR:
        case ACTIVATION_SIGMOID:
        case ACTIVATION_TANH:
            return 1;
    }
    return 0;
}

// Propagates a layer forward, computing the output neurons.
void layer_forward(Layer *l, float *inputs) {
    for (int j = 0; j < l->output_size; j++) {
        float sum = l->biases[j];
        for (int i = 0; i < l->input_size; i++) {
            sum += inputs[i] * l->weights[i * l->output_size + j];
        }

        // Apply the correct activation function.
        switch (l->activation) {
            case ACTIVATION_RELU:    l->outputs[j] = ReLU(sum); break;
            case ACTIVATION_LINEAR:  l->outputs[j] = sum; break;
            case ACTIVATION_SIGMOID: l->outputs[j] = Sigmoid(sum); break;
            case ACTIVATION_TANH:    l->outputs[j] = Tanh(sum); break;
        }
    }
}

// Propagates the network forward, calling layer_forward() in a chain.
void forward(Network *network, float *input) {
    float *current = input;
    for (int i = 0; i < network->num_layers; i++) {
        layer_forward(&network->layers[i], current);
        current = network->layers[i].outputs;
    }
}

// Calculates the mean squared error between a list of outputs, and a target list of outputs.
float mean_squared_error(float *output, float *target, int output_size) {
    float sum = 0;
    for (int i = 0; i < output_size; i++) {
        float loss = output[i] - target[i];
        sum += loss * loss;
    }
    return sum / output_size;
}

// Calculates the derivative of the mean squared error function.
float mse_derivative(float output, float target, int output_size) {
    return (2.0f / output_size) * (output - target);
}

// Calculates the delta of an output neuron.
// This is the same as taking the derivative of the neuron's loss with respect to the neuron's pre-activation value.
float output_delta(float output, float target, int output_size, Activation activation) {
    float dloss_da = mse_derivative(output, target, output_size);
    float da_dz = activation_derivative(activation, output);
    return dloss_da * da_dz;
}

// Propagates a layer's deltas to the previous layer.
void propagate_delta(Layer *l, float *deltas, Layer *prev, float *prev_deltas) {
    for (int i = 0; i < l->input_size; i++) {
        float sum = 0.0f;
        for (int j = 0; j < l->output_size; j++) {
            sum += deltas[j] * l->weights[i * l->output_size + j];
        }
        prev_deltas[i] = sum * activation_derivative(prev->activation, prev->outputs[i]);
    }
}

// Function for computing the weight and bias gradients for a layer.
void compute_gradients(Layer *l, float *inputs, float *deltas,
                        float *weight_grads, float *bias_grads) {
    for (int j = 0; j < l->output_size; j++) {
        bias_grads[j] = deltas[j];
        for (int i = 0; i < l->input_size; i++) {
            weight_grads[i * l->output_size + j] = deltas[j] * inputs[i];
        }
    }
}

// Function for applying weight and bias gradients to a layer.
// learning_rate controls the amount each weight and bias is nudged.
void apply_gradients(Layer *l, float *weight_grads, float *bias_grads, float learning_rate) {
    size_t weight_count = (size_t)l->input_size * l->output_size;

    // Apply weight gradients.
    for (size_t i = 0; i < weight_count; i++) {
        l->weights[i] -= learning_rate * weight_grads[i];
    }

    // Apply bias gradients.
    for (int j = 0; j < l->output_size; j++) {
        l->biases[j] -= learning_rate * bias_grads[j];
    }
}

// Function for applying backpropagation to a netword from the given deltas.
void network_backward_from_deltas(Network *network, TrainingContext *ctx, float *input, float *output_deltas, float learning_rate) {
    int num_layers = network->num_layers;
    float **deltas = ctx->deltas;

    memcpy(deltas[num_layers - 1], output_deltas, sizeof(float) * network->layers[num_layers - 1].output_size);

    for (int l = num_layers - 1; l > 0; l--) {
        propagate_delta(&network->layers[l], deltas[l], &network->layers[l - 1], deltas[l - 1]);
    }

    for (int l = 0; l < num_layers; l++) {
        Layer *current = &network->layers[l];
        float *layer_input = (l == 0) ? input : network->layers[l - 1].outputs;
        compute_gradients(current, layer_input, deltas[l], ctx->weight_grads, ctx->bias_grads);
        apply_gradients(current, ctx->weight_grads, ctx->bias_grads, learning_rate);
    }
}


// Function for calculating deltas and applying backpropagation to a network.
// Takes in the network, training context, inputs, target outputs, and the learning rate.
void network_backward(Network *network, TrainingContext *ctx, float *input, float *target, float learning_rate) {
    Layer *last = &network->layers[network->num_layers - 1];
    float *output_deltas = ctx->weight_grads; // Reuse existing scratch space
    for (int j = 0; j < last->output_size; j++) {
        output_deltas[j] = output_delta(last->outputs[j], target[j], last->output_size, last->activation);
    }
    network_backward_from_deltas(network, ctx, input, output_deltas, learning_rate);
}

// Computes the overall loss of the network relative to a target output.
float network_loss(Network *network, float *target) {
    Layer *last = &network->layers[network->num_layers - 1];
    return mean_squared_error(last->outputs, target, last->output_size);
}