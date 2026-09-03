#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include "storage.h"

Network empty_network(void) {
    Network net;
    net.num_layers = 0;
    net.layers = NULL;
    return net;
}

int save_network(Network *net, const char *filename) {
    FILE *f = fopen(filename, "wb");
    if (!f) return SAVE_ERR_OPEN;

    uint16_t version = FORMAT_VERSION;
    if (fwrite(&version, sizeof(uint16_t), 1, f) != 1) {
        fclose(f);
        return SAVE_ERR_WRITE;
    }

    if (fwrite(&net->num_layers, sizeof(int), 1, f) != 1) {
        fclose(f);
        return SAVE_ERR_WRITE;
    }

    for (int i = 0; i < net->num_layers; i++) {
        Layer *l = &net->layers[i];
        size_t w_count = (size_t)l->input_size * l->output_size;
        uint8_t activation = (uint8_t)l->activation;

        if (fwrite(&l->input_size, sizeof(int), 1, f) != 1 ||
            fwrite(&l->output_size, sizeof(int), 1, f) != 1 ||
            fwrite(&activation, sizeof(uint8_t), 1, f) != 1 ||
            fwrite(l->weights, sizeof(float), w_count, f) != w_count ||
            fwrite(l->biases, sizeof(float), l->output_size, f) != (size_t)l->output_size) {
            fclose(f);
            return SAVE_ERR_WRITE;
        }
    }

    fclose(f);
    return SAVE_OK;
}

Network load_network(const char *filename) {
    FILE *f = fopen(filename, "rb");
    if (!f) {
        fprintf(stderr, "Error loading network from path %s: Failed to open file.\n", filename);
        return empty_network();
    }

    uint16_t version;
    if (fread(&version, sizeof(uint16_t), 1, f) != 1) {
        fclose(f);
        fprintf(stderr, "Error loading network from path %s: Malformed or corrupted save. (version)\n", filename);
        return empty_network();
    }

    if (version != FORMAT_VERSION) {
        fclose(f);
        fprintf(stderr, "Error loading network from path %s: Unsupported format version %u (expected %u).\n",
                filename, version, (unsigned)FORMAT_VERSION);
        return empty_network();
    }

    Network net;
    if (fread(&net.num_layers, sizeof(int), 1, f) != 1) {
        fclose(f);
        fprintf(stderr, "Error loading network from path %s: Malformed or corrupted save. (num_layers)\n", filename);
        return empty_network();
    }

    net.layers = malloc(sizeof(Layer) * net.num_layers);

    for (int i = 0; i < net.num_layers; i++) {
        int input_size, output_size;
        uint8_t activation;
        if (fread(&input_size, sizeof(int), 1, f) != 1 ||
            fread(&output_size, sizeof(int), 1, f) != 1 ||
            fread(&activation, sizeof(uint8_t), 1, f) != 1 ||
            !activation_is_valid((Activation)activation)) {
            free(net.layers);
            fclose(f);
            fprintf(stderr, "Error loading network from path %s: Malformed or corrupted save. (input_size, output_size, & activation)\n", filename);
            return empty_network();
        }

        Layer l = create_layer(input_size, output_size, (Activation)activation);
        size_t w_count = (size_t)input_size * output_size;

        if (fread(l.weights, sizeof(float), w_count, f) != w_count ||
            fread(l.biases, sizeof(float), output_size, f) != (size_t)output_size) {
            free(l.weights);
            free(l.biases);
            free(l.outputs);
            for (int j = 0; j < i; j++) {
                free(net.layers[j].weights);
                free(net.layers[j].biases);
                free(net.layers[j].outputs);
            }
            free(net.layers);
            fclose(f);
            fprintf(stderr, "Error loading network from path %s: Malformed or corrupted save. (weights & biases)\n", filename);
            return empty_network();
        }

        net.layers[i] = l;
    }

    fclose(f);
    return net;
}