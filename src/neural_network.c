// Neural Network for the logical gate AND function


// This is a simple neural network that learns the logical AND function. This is using the sigmoid activation function. 

/**
 * Refences:
 * https://www.geeksforgeeks.org/implementation-of-artificial-neural-network-for-and-logic-gate-with-2-bit-binary-input/
 * https://cprimozic.net/blog/boolean-logic-with-neural-networks/
 * ChatGPT for propagate implementation and raylib for drawing the network.
 * https://www.youtube.com/watch?v=Sz-lXj0Ha6E
 * https://github.com/Frixoe/xor-neural-network/blob/master/XOR-Net-Notebook.ipynb
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <float.h>
#include "neural_network.h"
#include "raylib.h"

// Window for drawing the network.
#define WINDOW_WIDTH 650
#define WINDOW_HEIGHT 400

// The number of inputs, outputs, and hidden nodes.
#define NUM_INPUTS 2
#define NUM_OUTPUTS 1
#define NUM_HIDDEN 2

// A pattern is a set of inputs and outputs.
#define NUM_PATTERNS 4

// An epoch is a complete pass through the training patterns.
#define NUM_EPOCHS 10000

// The learning rate.
#define LEARNING_RATE 0.5

// The patience is the number of epochs to wait before stopping.
#define PATIENCE 10000

// The training patterns.
double inputs[NUM_PATTERNS][NUM_INPUTS] = {
  {0, 0},
  {0, 1},
  {1, 0},
  {1, 1}
};

// The desired outputs for each training pattern.
double outputs[NUM_PATTERNS][NUM_OUTPUTS] = {
  {0},
  {0},
  {0},
  {1}
};

/* The weights and biases. */
double hidden[NUM_HIDDEN];
double hidden_weights[NUM_INPUTS][NUM_HIDDEN];
double output_weights[NUM_HIDDEN][NUM_OUTPUTS];
double hidden_biases[NUM_HIDDEN];
double output_bias[NUM_OUTPUTS];
double output[NUM_OUTPUTS];

/**
 * The sigmoid function.
 * @param x The input.
 * @return The output.
 */
double sigmoid(double x) {
  return 1 / (1 + exp(-x));
}

/**
 * The derivative of the sigmoid function.
 * @param x The input.
 * @return The output.
 */
double sigmoid_derivative(double x) {
  return x * (1 - x);
}

/**
 * Initialize the weights.
 */
void initialize_weights(void) {
  for (int i = 0; i < NUM_INPUTS; i++) {
    for (int j = 0; j < NUM_HIDDEN; j++) {
      // Initialize the weights to a random value between 0 and 1.
      hidden_weights[i][j] = (double) rand() / (double) RAND_MAX;
    }
  }
  for (int i = 0; i < NUM_HIDDEN; i++) {
    for (int j = 0; j < NUM_OUTPUTS; j++) {
      // Initialize the weights to a random value between 0 and 1.
      output_weights[i][j] = (double) rand() / (double) RAND_MAX;
    }
  }
}

/**
 * Initialize the biases.
 */
void initialize_biases(void) 
{
  // Initialize the biases to a random value between 0 and 1.
  for (int i = 0; i < NUM_HIDDEN; i++) {
    hidden_biases[i] = (double) rand() / (double) RAND_MAX;
  }
  // Initialize the biases to a random value between 0 and 1.
  for (int i = 0; i < NUM_OUTPUTS; i++) {
    output_bias[i] = (double) rand() / (double) RAND_MAX;
  }
}

/**
 * Forward propagate the inputs.
 * @param p The pattern index.
 */
void forward_propagate(int p) {
  // Calculate the outputs of the hidden layer.
  for (int i = 0; i < NUM_HIDDEN; i++) {
    hidden[i] = 0.0;
    // Calculate the weighted sum of the inputs and the weights.
    for (int j = 0; j < NUM_INPUTS; j++) {
      hidden[i] += inputs[p][j] * hidden_weights[j][i];
    }
    // Add the bias.
    hidden[i] += hidden_biases[i];
    hidden[i] = sigmoid(hidden[i]);
  }
  // Calculate the outputs of the output layer.
  for (int i = 0; i < NUM_OUTPUTS; i++) {
    output[i] = 0.0;
    // Calculate the weighted sum of the hidden layer outputs and the weights.
    for (int j = 0; j < NUM_HIDDEN; j++) {
      output[i] += hidden[j] * output_weights[j][i];
    }
    // Add the bias.
    output[i] += output_bias[i];
    output[i] = sigmoid(output[i]);
  }
}

/**
 * Back propagate the errors.
 * @param p The pattern index.
 */
void back_propagate(int p) {
  double error;
  // Update the weights and biases for the output layer.
  for (int i = 0; i < NUM_OUTPUTS; i++) {
    error = outputs[p][i] - output[i];
    for (int j = 0; j < NUM_HIDDEN; j++) {
      // introduce a learning rate to the weights.  The learning rate is primarily used during the weight and bias update step in the backpropagation algorithm. 
      output_weights[j][i] += LEARNING_RATE * error * hidden[j] * sigmoid_derivative(output[i]);
    }
    output_bias[i] += error * sigmoid_derivative(output[i]);
  }
  // Update the weights and biases for the hidden layer.
  for (int i = 0; i < NUM_HIDDEN; i++) {
    error = 0.0;
    // Calculate the error.
    for (int j = 0; j < NUM_OUTPUTS; j++) {
      error += output_weights[i][j] * (outputs[p][j] - output[j]);
    }
    // Update the weights and biases.
    for (int j = 0; j < NUM_INPUTS; j++) {
      hidden_weights[j][i] += error * inputs[p][j] * sigmoid_derivative(hidden[i]);
    }
    hidden_biases[i] += LEARNING_RATE * error * sigmoid_derivative(hidden[i]);
  }
}

// Use raylib to draw the network.
void draw_network(void) {
  // Display the epoch and the learning rate in the top left corner of the screen. Put them together like Epoch , Learning Rate

  // Draw the input layer.
  for (int i = 0; i < NUM_INPUTS; i++) {
    // These numbers are arbitrary to where the circles are drawn.
    DrawCircle(100, 100 + i * 100, 30, RED);
  }
  // Draw the hidden layer.
  for (int i = 0; i < NUM_HIDDEN; i++) {
    DrawCircle(300, 100 + i * 100, 30, BLUE);
  }
  // Draw the output layer.
  for (int i = 0; i < NUM_OUTPUTS; i++) {
    DrawCircle(500, 100 + i * 100, 30, GREEN);
  }
  // Draw the weights.
  for (int i = 0; i < NUM_INPUTS; i++) {
    for (int j = 0; j < NUM_HIDDEN; j++) {
      DrawLine(100 + 30, 100 + i * 100, 300 - 30, 100 + j * 100, GRAY);
    }
  }
  for (int i = 0; i < NUM_HIDDEN; i++) {
    for (int j = 0; j < NUM_OUTPUTS; j++) {
      DrawLine(300 + 30, 100 + i * 100, 500 - 30, 100 + j * 100, GRAY);
    }
  }

  // Draw labels or text annotations.
  DrawText("Input Layer", 80, 60, 20, BLACK);
  DrawText("Hidden Layer", 280, 60, 20, BLACK);
  DrawText("Output Layer", 480, 60, 20, BLACK);
}

int calculate_loss(int p) {
  double loss = 0.0;
  for (int i = 0; i < NUM_OUTPUTS; i++) {
    loss += pow(outputs[p][i] - output[i], 2);
  }
  return loss;
}

/**
 * Train the network.
 */
int main(void) {
  initialize_weights();
  initialize_biases();

  // Draw the network.
  InitWindow(WINDOW_WIDTH, WINDOW_HEIGHT, "Neural Network");
  SetTargetFPS(60);

  int epoch = 0;
  double best_validation_loss = DBL_MAX;
  int patience_counter = 0;
  int running = 1;
  while (!WindowShouldClose()) {
    BeginDrawing();
    ClearBackground(WHITE);
    draw_network();
    EndDrawing();

    // Neural network should stop when the epoch is reached.

    // Train the network but update the epoch global variable. 
    if (running) {
      for (int e = epoch; e < NUM_EPOCHS; e++) {
        
        for (int p = 0; p < NUM_PATTERNS; p++) {
          forward_propagate(p);
          back_propagate(p);
        }

        // Update text with the epoch (e) / NUM_EPOCHS in the top left corner of the screen. 
        TraceLog(LOG_INFO, "Epoch: %d / %d", e + 1, NUM_EPOCHS);
        
        double validation_loss = 0.0;
        for (int p = 0; p < NUM_PATTERNS; p++) {
          forward_propagate(p);
          validation_loss += calculate_loss(p);
        }
        validation_loss /= NUM_PATTERNS;

        if (validation_loss < best_validation_loss) {
          best_validation_loss = validation_loss;
          patience_counter = 0;
        } else {
          patience_counter++;
          if (patience_counter >= PATIENCE-1) {
            break;
          }
        }
        epoch++;
      }
      running = 0;
    }

    DrawText(TextFormat("Epoch: %d / %d", epoch + 1, NUM_EPOCHS), 10, 10, 20, BLACK);
    

    for (int p = 0; p < NUM_PATTERNS; p++) {
      forward_propagate(p);
      // Draw the inputs and outputs. Put them at the bottom of the screen. Put them below the network.
      DrawText(TextFormat("Input: %d %d Output: %lf", (int)inputs[p][0], (int)inputs[p][1], output[0]), 10, 300 + p * 20, 20, BLACK);
    }

  }

  CloseWindow();
  return 0;

}
