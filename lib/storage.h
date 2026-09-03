#ifndef PNNL_STORAGE_H
#define PNNL_STORAGE_H

#include <stdint.h>
#include "neuralnet.h"

#define SAVE_OK        0
#define SAVE_ERR_OPEN  1
#define SAVE_ERR_WRITE 2

// Save format changelog:
//   v1 - initial format: version, num_layers, then per layer:
//        (input_size, output_size, activation as int, weights, biases)
//   v2 - activation stored as uint8_t instead of int (4 bytes -> 1 byte)
//   v3 - added ACTIVATION_SIGMOID and ACTIVATION_TANH
#define FORMAT_VERSION 3

Network empty_network(void);
int save_network(Network *net, const char *filename);
Network load_network(const char *filename);

#endif // PNNL_STORAGE_H