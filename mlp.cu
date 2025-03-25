#include <iostream>
#include <cstdlib>
#include <cuda_runtime.h>

#define IDX(i, j, cols) ((i) * (cols) + (j))

__global__ void matmul(float *A, float *B, float *C, int M, int N, int P) {
    unsigned int row = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < P) {
        float sum = 0.0;
        for (int k = 0; k < N; k++) {
            sum += A[IDX(row, k, N)] * B[IDX(k, col, P)];
        }
        C[IDX(row, col, P)] = sum;
    }
}

__global__ void relu(float *X, int size) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size)
        X[idx] = fmaxf(0.0, X[idx]);
}

__global__ void relu_backward(float *dA, float *Z, int size) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size)
        dA[idx] *= (Z[idx] > 0);
}

__global__ void mse_loss(float *pred, float *target, float *loss, int size) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size)
        atomicAdd(loss, (pred[idx] - target[idx]) * (pred[idx] - target[idx]) / size);
}

__global__ void mse_backward(float *pred, float *target, float *grad, int size) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size)
        grad[idx] = 2.0f * (pred[idx] - target[idx]) / size;
}

__global__ void update_weights(float *W, float *dW, int size, float lr) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size)
        W[idx] -= lr * dW[idx];
}

void train_mlp(float *X, float *Y, int input_dim, int hidden_dim, int output_dim, int batch_size, int epochs, float lr) {
  
    float *d_X, *d_Y, *W1, *b1, *W2, *b2, *Z1, *A1, *Z2, *A2;
    float *dW1, *db1, *dW2, *db2, *dZ2, *dA1, *dZ1;

    cudaMalloc((void**)&d_X, batch_size * input_dim * sizeof(float));
    cudaMalloc((void**)&d_Y, batch_size * output_dim * sizeof(float));
    cudaMalloc((void**)&W1, input_dim * hidden_dim * sizeof(float));
    cudaMalloc((void**)&b1, hidden_dim * sizeof(float));
    cudaMalloc((void**)&W2, hidden_dim * output_dim * sizeof(float));
    cudaMalloc((void**)&b2, output_dim * sizeof(float));
    cudaMalloc((void**)&Z1, batch_size * hidden_dim * sizeof(float));
    cudaMalloc((void**)&A1, batch_size * hidden_dim * sizeof(float));
    cudaMalloc((void**)&Z2, batch_size * output_dim * sizeof(float));
    cudaMalloc((void**)&A2, batch_size * output_dim * sizeof(float));
    cudaMalloc((void**)&dW1, input_dim * hidden_dim * sizeof(float));
    cudaMalloc((void**)&db1, hidden_dim * sizeof(float));
    cudaMalloc((void**)&dW2, hidden_dim * output_dim * sizeof(float));
    cudaMalloc((void**)&db2, output_dim * sizeof(float));
    cudaMalloc((void**)&dZ2, batch_size * output_dim * sizeof(float));
    cudaMalloc((void**)&dA1, batch_size * hidden_dim * sizeof(float));
    cudaMalloc((void**)&dZ1, batch_size * hidden_dim * sizeof(float));

    // Initialize weights randomly
    float *h_W1 = (float *)malloc(input_dim * hidden_dim * sizeof(float));
    float *h_W2 = (float *)malloc(hidden_dim * output_dim * sizeof(float));

    for (int i = 0; i < input_dim * hidden_dim; i++) h_W1[i] = ((float)rand() / RAND_MAX) * 0.01;
    for (int i = 0; i < hidden_dim * output_dim; i++) h_W2[i] = ((float)rand() / RAND_MAX) * 0.01;

    cudaMemcpy(W1, h_W1, input_dim * hidden_dim * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(W2, h_W2, hidden_dim * output_dim * sizeof(float), cudaMemcpyHostToDevice);

    dim3 block(16, 16);
    dim3 grid((hidden_dim + 15) / 16, (batch_size + 15) / 16);

    for (int epoch = 0; epoch < epochs; epoch++) {
        // fwd pass
        matmul<<<grid, block>>>(d_X, W1, Z1, batch_size, input_dim, hidden_dim);
        relu<<<(batch_size * hidden_dim + 255) / 256, 256>>>(Z1, batch_size * hidden_dim);
        
        matmul<<<grid, block>>>(Z1, W2, Z2, batch_size, hidden_dim, output_dim);

        // loss
        float *loss;
        cudaMalloc((void**)&loss, sizeof(float));
        cudaMemset(loss, 0, sizeof(float));
        mse_loss<<<(batch_size * output_dim + 255) / 256, 256>>>(Z2, d_Y, loss, batch_size * output_dim);
        
        float h_loss;
        cudaMemcpy(&h_loss, loss, sizeof(float), cudaMemcpyDeviceToHost);
        std::cout << "Epoch " << epoch << " Loss: " << h_loss << std::endl;

        // bwd
        mse_backward<<<(batch_size * output_dim + 255) / 256, 256>>>(Z2, d_Y, dZ2, batch_size * output_dim);
        matmul<<<grid, block>>>(Z1, dZ2, dW2, hidden_dim, batch_size, output_dim);
        
        relu_backward<<<(batch_size * hidden_dim + 255) / 256, 256>>>(dA1, Z1, batch_size * hidden_dim);
        matmul<<<grid, block>>>(d_X, dA1, dW1, input_dim, batch_size, hidden_dim);

        // update
        update_weights<<<(input_dim * hidden_dim + 255) / 256, 256>>>(W1, dW1, input_dim * hidden_dim, lr);
        update_weights<<<(hidden_dim * output_dim + 255) / 256, 256>>>(W2, dW2, hidden_dim * output_dim, lr);
    }

    cudaFree(d_X); cudaFree(d_Y); cudaFree(W1); cudaFree(W2); cudaFree(Z1); cudaFree(A1);
}

