#ifndef PNNL_NEURALNET_H
#define PNNL_NEURALNET_H

#ifndef M_PI
#define M_PI 3.14159265358979323846f
#endif // M_PI

typedef enum {
    ACTIVATION_RELU,
    ACTIVATION_LINEAR,
    ACTIVATION_SIGMOID,
    ACTIVATION_TANH
} Activation;

typedef struct {
    int input_size;
    int output_size;
    float *weights;
    float *biases;
    float *outputs;
    Activation activation;
} Layer;

typedef struct {
    Layer *layers;
    int num_layers;
} Network;

typedef struct {
    float **deltas;
    float *weight_grads;
    float *bias_grads;
} TrainingContext;

float random_gaussian(float mean, float stddev);
float random_uniform(float limit);

Layer create_layer(int input_size, int output_size, Activation activation);
Network create_network(int num_layers, Layer *layers);
Layer clone_layer(Layer *l);
Network clone_network(Network *net);
void free_layer(Layer *l);
void free_network(Network *net);
void mutate_layer(Layer *l, float mutation_rate, float mutation_strength);
void mutate_network(Network *net, float mutation_rate, float mutation_strength);

TrainingContext create_training_context(Network *network);
void free_training_context(TrainingContext *ctx, int num_layers);

float ReLU(float x);
float ReLU_derivative(float x);
float Sigmoid(float x);
float Sigmoid_derivative(float output);
float Tanh(float x);
float Tanh_derivative(float output);
float activation_derivative(Activation activation, float output);
int activation_is_valid(Activation activation);

void layer_forward(Layer *l, float *inputs);
void forward(Network *network, float *input);

float mean_squared_error(float *output, float *target, int output_size);
float mse_derivative(float output, float target, int output_size);

float output_delta(float output, float target, int output_size, Activation activation);
void propagate_delta(Layer *l, float *deltas, Layer *prev, float *prev_deltas);

void compute_gradients(Layer *l, float *inputs, float *deltas, float *weight_grads, float *bias_grads);
void apply_gradients(Layer *l, float *weight_grads, float *bias_grads, float learning_rate);
void network_backward_from_deltas(Network *network, TrainingContext *ctx, float *input, float *output_deltas, float learning_rate);
void network_backward(Network *network, TrainingContext *ctx, float *input, float *target, float learning_rate);

float network_loss(Network *network, float *target);

#endif // PNNL_NEURALNET_H