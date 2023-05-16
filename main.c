#include <stdio.h>
#include <math.h>
#include <time.h>
#include <stdlib.h>
#include <stdbool.h>


struct LinearParams {
    double w1;
    double w0;
};

double random_param();
double predict(struct LinearParams params, double x);
double compute_loss(struct LinearParams params, double x, double y, bool derivative);

/**
 * Simple linear regression.
 * 
 * Optimize the equation with parameters w0 and w1:
 * y = w1 * x + w0
*/
int main() {
    double train_data[][2] = {
        {1.0f, 2.0f},
        {2.0f, 4.0f},
        {3.0f, 6.0f},
        {4.0f, 8.0f},
        {5.0f, 10.0f},
        {6.0f, 12.0f},
        {7.0f, 14.0f},
        {8.0f, 16.0f},
        {9.0f, 18.0f},
    };
    int train_data_size = sizeof(train_data) / sizeof(train_data[0]);
    srand(time(NULL));

    // Initialize weights
    struct LinearParams params;

    params.w1 = random_param();
    params.w0 = 0;

    // Training loop
    for (int i = 0; i < 100; i++) {
        double loss = 0;
        double w1_derivative = 0;
        double w0_derivative = 0;

        for (int j = 0; j < train_data_size; j++) {
            double x = train_data[j][0];
            double y = train_data[j][1];

            loss += compute_loss(params, x, y, false);
            w1_derivative += compute_loss(params, x, y, true) * x;
            w0_derivative += compute_loss(params, x, y, true);

            params.w1 -= 0.02 * w1_derivative;
            params.w0 -= 0.02 * w0_derivative;
        }

        // printf("Epoch: %i Loss: %f\n", i, loss);
    }

    // printf("%f\n", params.w0);
    // printf("%f\n", params.w1);

    printf("%f\n", predict(params, 235));
}

double predict(struct LinearParams params, double x) {
    return params.w0 + params.w1 * x;
}

double compute_loss(struct LinearParams params, double x, double y, bool derivative) {
    double y_hat = predict(params, x);

    if (derivative) {
        return 2 * (y_hat - y);
    }

    return pow(y_hat - y, 2);
}

double random_param() {
    return (double)rand() / (double)RAND_MAX;
}
