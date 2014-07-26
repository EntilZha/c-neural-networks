//
//  main.cpp
//  NeuralNetworks
//
//  Created by Pedro Rodriguez on 4/17/14.
//  Copyright (c) 2014 cs189. All rights reserved.
//

#include <iostream>
#include <math.h>
#include <algorithm>
#include <fstream>
#include <sstream>
#include <string>
#include <unistd.h>
#include <vector>
#include <stdlib.h>
#include <cstdlib>
#include <stddef.h>
#include <stdio.h>
#include <string.h>
#include <stdexcept>
#include <omp.h>
#include <time.h>
#include <cblas.h>
using namespace std;

int num_digits = 10;
int chunk_size = 200;
int N = 60000;
int features = 784;
int epochs = 500;

double sigmoid(double x) {
    double v = 1 / (1 + exp(-x));
    return v;
}

double mse(double* x_point, double* weights, double* bias, int* labels) {
    double sum = 0;
    for (int k = 0; k < num_digits; k++) {
        double tk = (k == labels[0]) ? 1.0 : 0.0;
        double v = sigmoid(cblas_ddot(features, x_point, 1, weights + k * features, 1) + bias[k]);
        sum += .5 * pow(v - tk, 2);
    }
    return sum;
}

double finite_difference(double* x, double* weights, double* bias, double epsilon, int* labels, int choose_j) {
    double* w_plus = (double*) malloc(sizeof(double) * features * num_digits);
    for (int k = 0; k < num_digits; k++) {
        for (int j = 0; j < features; j++) {
            if (j == choose_j) {
                w_plus[k * features + j] = epsilon + weights[k * features + j];
            } else {
                w_plus[k * features + j] = weights[k * features + j];
            }
        }
    }
    double f_w = mse(x, weights, bias, labels);
    double f_w_plus = mse(x, w_plus, bias, labels);
    return (f_w_plus - f_w) / epsilon;
}

void print_output(double* y) {
    for (int i = 0; i < 200; i++) {
        printf("%f\n", y[i]);
    }
}

double rand_double() {
    return ((double) rand() / (RAND_MAX))*2-1;
}

void print_data_point(double* data_point) {
    for (int i = 0; i < 28; i++) {
        for (int j = 0; j < 28; j++) {
            printf("%d ", data_point[i * 28 + j] > 0);
        }
        printf("\n");
    }
}

void print_data_row(double* data_point) {
    for (int i = 0; i < features; i++) {
        printf("%.0f ", data_point[i]);
    }
    printf("\n");
}

void read_all_data(double* x_vals, int* t_vals, char* filename) {
    string line;
    ifstream data(filename);
    if (!data.is_open()) {
        throw invalid_argument("Could not read input file");
    }
    int j = 0;
    int k = 0;
    while (getline(data, line)) {
        istringstream ss(line);
        string token;
        int i = 0;
        while (getline(ss, token, ',')) {
            char token_array[4];
            token_array[0] = token[0];
            token_array[1] = token[1];
            token_array[2] = token[2];
            token_array[3] = '\0';
            if (i == features) {
                t_vals[k] = atoi(token_array);
                i++;
                k++;
            } else {
                x_vals[j * features + i] = atof(token_array);
                i++;
            }
        }
        j++;
    }
    data.close();
    return;
}

double eta(int epoch) {
    //Exponential going through (0, 1), (2500, .5), (5000, .1)
    //return .1 / pow(epoch + 1, .5);
    return .01;
}

void forward_propagate(double* x, double* y, double* weights, double* bias, int num_elements) {
    //Dot product x0 with weights and add bk to it.
    for (int i = 0; i < num_elements; i++) {
        for (int k = 0; k < num_digits; k++) {
            double result = bias[k] + cblas_ddot(features, x + i * features, 1, weights + k * features, 1);
            y[i * num_digits + k] = sigmoid(result);
        }
    }
    return;
}

void backward_propagate(double* x, double* y, int* true_vals, double* weights, double* bias, int num_elements, int epoch) {
    for (int i = 0; i < num_elements; i++) {
        for (int k = 0; k < num_digits; k++) {
            double yk = y[i * num_digits + k];
            double tk = (k == true_vals[i]) ? 1.0 : 0.0;
            double coef = -1 * eta(epoch) * yk * (1 - yk) * (yk - tk);
            cblas_daxpy(features, coef, x + i * features, 1, weights + k * features, 1);
            bias[k] += coef;
        }
    }
    return;
}

void backward_propagate_cross_entropy(double* x, double* y, int* true_vals, double* weights, double* bias, int num_elements, int epoch) {
    for (int i = 0; i < num_elements; i++) {
        for (int k = 0; k < num_digits; k++) {
            double yk = y[i * num_digits + k];
            double tk = (k == true_vals[i]) ? 1.0 : 0.0;
            double coef = -1 * eta(epoch) * (yk - tk);
            cblas_daxpy(features, coef, x + i * features, 1, weights + k * features, 1);
            bias[k] += coef;
        }
    }
    return;
}

double calculate_error(double* x, double* weights, double* bias, int* true_vals, int N_error) {
    //Iterate through each data point
    int* predictions = (int*) calloc(num_digits, sizeof(int));
    int* correct_predictions = (int*) calloc(num_digits, sizeof(int));
    double errors = 0;
    for (int i = 0; i < N_error; i++) {
        //Iterate through each possible digit, take the max
        double max_val = 0;
        int max_digit = 0;
        for (int k = 0; k < num_digits; k++) {
            double val = bias[k];
            for (int j = 0; j < features; j++) {
                val += x[i * features + j] * weights[k * features + j];
            }
            val = sigmoid(val);
            //printf("d=%d v=%.4f | ", k, val);
            if (val >= max_val) {
                max_val = val;
                max_digit = k;
            }
        }
        //printf("\nTrue: %d Pred: %d\n", true_vals[i], max_digit);
        //print_data_point(x + i * features);
        predictions[max_digit] += 1;
        if (max_digit != true_vals[i]) {

            errors += 1;
        } else {
            correct_predictions[max_digit] += 1;
        }
    }
    for (int i = 0; i < num_digits; i++) {
        //printf("%d: %d/%d \t", i, correct_predictions[i], predictions[i]);
    }
    free(predictions);
    return errors / N_error;
}

void normalize_data(double* x, int size) {
    for (int i = 0; i < size; i++) {
        double norm = 1.0 / 255;
        cblas_dscal(features, norm, x + i * features, 1);
    }
}

void shuffle_data(double* x, int* t, int length) {
    //Prepare framework to do random shuffles
    double* x_copy = (double*) malloc(sizeof(double) * length * features);
    int* t_copy = (int*) malloc(sizeof(int) * length);
    vector<int> order;
    for (int i = 0; i < length; ++i) order.push_back(i);
    random_shuffle(order.begin(), order.end());
    int i = 0;
    int k;
    for (std::vector<int>::iterator it=order.begin(); it!=order.end(); ++it) {
        k = *it;
        for (int j = 0; j < features; j++) {
            x_copy[i * features + j] = x[k * features + j];
        }
        t_copy[i] = t[k];
        i++;
    }
    for (int i = 0; i < length; i++) {
        for (int j = 0; j < features; j++) {
            x[i * features + j] = x_copy[i * features + j];
        }
        t[i] = t_copy[i];
    }
    free(x_copy);
    free(t_copy);
}

int main(int argc, const char * argv[])
{
    //srand (1);
    double* weights = (double*) malloc(sizeof(double) * num_digits * features);
    double* bias = (double*) malloc(sizeof(double) * num_digits);
    for (int i = 0; i < num_digits; i++) {

        bias[i] = rand_double();
    }
    for (int i = 0; i < features * num_digits; i++) {
        weights[i] = rand_double();
    }
    //Holds number label values for each of N data points
    int* true_values = (int*) malloc(sizeof(int) * N);
    //Holds the initial x values, 784*N
    double* x = (double*) malloc(sizeof(double) * N * features);
    double* x_test = (double*) malloc(sizeof(double) * 10000 * features);
    int* t_test = (int*) malloc(sizeof(int) * 10000);
    //Holds 200 of the N data points output values
    double* y = (double*) malloc(sizeof(double) * chunk_size * num_digits);
    char training_file[] = "../train-full.txt";
    char test_file[] = "../dataset-10000.txt";
    read_all_data(x, true_values, training_file);
    read_all_data(x_test, t_test, test_file);
    normalize_data(x_test, 10000);
    normalize_data(x, N);
    shuffle_data(x, true_values, N);
    clock_t start, end, forward_start, forward_end, backward_start, backward_end;
    start = clock();
    int N_test = 10000;
    for (int e = 1; e <= epochs; e++) {
        for (int i = 0; i < N; i += chunk_size) {
            int num_elements = min(chunk_size, N - i);
            forward_propagate(x + i * features, y, weights, bias, num_elements);
            // backward_propagate(x + i * features, y, true_values + i, weights, bias, num_elements, e);
            backward_propagate_cross_entropy(x + i * features, y, true_values + i, weights, bias, num_elements, e);
        }
        if (e % 10 == 0 || e < 10) {
            //printf("Epoch %d\n", e);
            //printf("\nTraining Error: %f\n", calculate_error(x, weights, bias, true_values, N));
            //printf("\nTest Error: %f\n", calculate_error(x_test, weights, bias, t_test, 10000));
            printf("%d,%f,%f\n", e,
                    calculate_error(x, weights, bias, true_values, N),
                    calculate_error(x_test, weights, bias, t_test, 10000));
        }
        shuffle_data(x, true_values, N);
    }
    end = clock();
    double diff = (double) (end - start) / CLOCKS_PER_SEC;

    printf("Running Time: %f\n", diff);
    free(x);
    free(weights);
    free(bias);
    free(true_values);
    free(y);
    return 0;
}
