#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <time.h>
#include "neuralnet.h"
#include "storage.h"

// Holds the MNIST images and labels once loaded into memory.
typedef struct {
    float *images;    // flattened: num_images * image_size, normalized to 0.0-1.0
    uint8_t *labels;  // num_images, values 0-9
    int num_images;
    int image_size;   // rows * cols, e.g. 784 for 28x28
} MnistData;

// Reads a big-endian 32-bit integer, since IDX files store ints MSB-first,
// while most consumer hardware is little-endian.
uint32_t read_be_uint32(FILE *f) {
    unsigned char bytes[4];
    fread(bytes, 1, 4, f);
    return ((uint32_t)bytes[0] << 24) | ((uint32_t)bytes[1] << 16) |
           ((uint32_t)bytes[2] << 8)  | (uint32_t)bytes[3];
}

// Loads an MNIST image/label pair from their raw idx-ubyte files.
MnistData load_mnist(const char *images_path, const char *labels_path) {
    MnistData data = {0};

    FILE *img_f = fopen(images_path, "rb");
    FILE *lbl_f = fopen(labels_path, "rb");
    if (!img_f || !lbl_f) {
        fprintf(stderr, "Failed to open MNIST files (%s, %s)\n", images_path, labels_path);
        exit(1);
    }

    uint32_t img_magic = read_be_uint32(img_f);
    uint32_t num_images = read_be_uint32(img_f);
    uint32_t rows = read_be_uint32(img_f);
    uint32_t cols = read_be_uint32(img_f);

    uint32_t lbl_magic = read_be_uint32(lbl_f);
    uint32_t num_labels = read_be_uint32(lbl_f);

    if (img_magic != 0x00000803 || lbl_magic != 0x00000801) {
        fprintf(stderr, "Invalid MNIST file magic number\n");
        exit(1);
    }
    if (num_images != num_labels) {
        fprintf(stderr, "Image/label count mismatch (%u vs %u)\n", num_images, num_labels);
        exit(1);
    }

    data.num_images = (int)num_images;
    data.image_size = (int)(rows * cols);
    data.images = malloc(sizeof(float) * (size_t)data.num_images * data.image_size);
    data.labels = malloc(sizeof(uint8_t) * data.num_images);

    unsigned char *raw_pixels = malloc(data.image_size);
    for (int i = 0; i < data.num_images; i++) {
        fread(raw_pixels, 1, data.image_size, img_f);
        for (int p = 0; p < data.image_size; p++) {
            data.images[(size_t)i * data.image_size + p] = raw_pixels[p] / 255.0f;
        }
    }
    free(raw_pixels);

    fread(data.labels, 1, data.num_images, lbl_f);

    fclose(img_f);
    fclose(lbl_f);
    return data;
}

// Frees the memory allocated by load_mnist.
void free_mnist(MnistData *data) {
    free(data->images);
    free(data->labels);
}

// Converts a single digit label (0-9) into a one-hot target vector.
void one_hot(uint8_t label, float *target, int num_classes) {
    for (int i = 0; i < num_classes; i++) {
        target[i] = (i == label) ? 1.0f : 0.0f;
    }
}

// Returns the index of the highest-value output neuron, i.e. the network's predicted digit.
int predict(Network *net) {
    Layer *last = &net->layers[net->num_layers - 1];
    int best = 0;
    for (int i = 1; i < last->output_size; i++) {
        if (last->outputs[i] > last->outputs[best]) best = i;
    }
    return best;
}

// Runs the network over a dataset and returns the fraction of correct predictions (0.0-1.0).
float evaluate_accuracy(Network *net, MnistData *data) {
    int correct = 0;
    for (int i = 0; i < data->num_images; i++) {
        float *input = &data->images[(size_t)i * data->image_size];
        forward(net, input);
        if (predict(net) == data->labels[i]) {
            correct++;
        }
    }
    return (float)correct / data->num_images;
}

int main(int argc, char *argv[]) {
    srand((unsigned)time(NULL)); // Sets the random seed to be time dependant.

    printf("Loading MNIST data...\n");
    MnistData train_data = load_mnist("MNIST/train-images-idx3-ubyte", "MNIST/train-labels-idx1-ubyte");
    MnistData test_data = load_mnist("MNIST/t10k-images-idx3-ubyte", "MNIST/t10k-labels-idx1-ubyte");
    printf("Loaded %d training images, %d test images (%d pixels each)\n",
           train_data.num_images, test_data.num_images, train_data.image_size);
    
    Network net;
    if (argc > 1) {
        net = load_network(argv[1]);
        if (net.layers == NULL) {
            fprintf(stderr, "Error loading MNIST model!\n");
            return 1;
        }

        printf("Loaded pre-existing model from path: %s\n", argv[1]);
    } else {
        // Build the network: 784 -> 128 -> 64 -> 10
        Layer layers[3];
        layers[0] = create_layer(train_data.image_size, 128, ACTIVATION_SIGMOID);
        layers[1] = create_layer(128, 64, ACTIVATION_SIGMOID);
        layers[2] = create_layer(64, 10, ACTIVATION_LINEAR);
        net = create_network(3, layers);
    }

    TrainingContext ctx = create_training_context(&net);

    float target[10];
    float learning_rate = 0.30f;
    int epochs = 10;

    for (int epoch = 1; epoch <= epochs; epoch++) {
        float total_loss = 0.0f;

        for (int i = 0; i < train_data.num_images; i++) {
            float *input = &train_data.images[(size_t)i * train_data.image_size];
            one_hot(train_data.labels[i], target, 10);

            forward(&net, input);
            total_loss += network_loss(&net, target);
            network_backward(&net, &ctx, input, target, learning_rate);

            if (i % (int)(train_data.num_images / 10) == 0) {
                printf("- Trained %d images out of %d. %.2f%% done.\n", 
                    i+1, train_data.num_images, (float)(i+1) / (float)train_data.num_images * 100.0f);
            }
        }

        float train_accuracy = evaluate_accuracy(&net, &train_data);
        printf("Epoch %d/%d: avg loss = %f, train accuracy = %.2f%%\n",
               epoch, epochs, total_loss / train_data.num_images, train_accuracy * 100.0f);
    }

    float test_accuracy = evaluate_accuracy(&net, &test_data);
    printf("Final test accuracy: %.2f%%\n", test_accuracy * 100.0f);

    // Save the trained network for later use.
    int save_result = save_network(&net, "mnist_model.bin");
    if (save_result == SAVE_OK) {
        printf("Saved trained network to mnist_model.bin\n");
    } else {
        fprintf(stderr, "Failed to save network (error %d)\n", save_result);
    }

    free_training_context(&ctx, net.num_layers);
    free_mnist(&train_data);
    free_mnist(&test_data);

    return 0;
}