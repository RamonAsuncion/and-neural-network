#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <float.h>
#include "neural_network.h"
#include "raylib.h"
#include <pthread.h>
#include <unistd.h>

// Window for drawing the network.
#define WINDOW_WIDTH 650
#define WINDOW_HEIGHT 400

// The number of inputs, outputs, and hidden nodes.
#define NUM_INPUTS 2
#define NUM_OUTPUTS 1
#define NUM_HIDDEN 1

// A pattern is a set of inputs and outputs.
#define NUM_PATTERNS 4

// An epoch is a complete pass through the training patterns.
#define NUM_EPOCHS 5000

// The learning rate.
#define LEARNING_RATE 0.1

// The patience is the number of epochs to wait before stopping.
#define PATIENCE 5000

// The delay between epochs in microseconds. (Program)
#define DELAY 1000

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


volatile int running = 1;

int epoch = 0;

/**
 * The sigmoid function.
 * @param x The input.
 * @return The output.
 */
double sigmoid(double x) 
{
  return 1 / (1 + exp(-x));
}

/**
 * The derivative of the sigmoid function.
 * @param x The input.
 * @return The output.
 */
double sigmoid_derivative(double x) 
{
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
void forward_propagate(int p) 
{
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
void back_propagate(int p) 
{
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

// The brightest lines (i.e., the lines with the highest absolute weights) are the most used connections in the neural network. Conversely, the darker lines (i.e., the lines with the lowest absolute weights) are the least used connections
Color weight_to_color(double weight) 
{
  // Map the weight to a color. For example, we could use a simple linear mapping
  // from [-1, 1] to [0, 1] and then map this to the colors grey and orange.
  double value = (weight + 1.0) / 2.0;
  int r = (int)(255 * value);
  int g = (int)(255 * (1 - value));
  int b = 0;
  return (Color){r, g, b, 255};
}


int is_neuron_used(int i) 
{
  // Return 1 if the neuron is used, otherwise return 0.
  for (int j = 0; j < NUM_HIDDEN; j++) {
    if (hidden_weights[i][j] != 0) {
      return 1;
    }
  }
  return 0;
}

int is_weight_used(int i, int j) 
{
  // Return 1 if the weight is used, otherwise return 0.
  if (hidden_weights[i][j] != 0) {
    return 1;
  }
  return 0;
}

void draw_network(double *activations) 
{
  // Draw the input layer.
  for (int i = 0; i < NUM_INPUTS; i++) {
    // Use the activation value to determine the color.
    Color color = weight_to_color(activations[i]);
    DrawCircle(100, 100 + i * 100, 30, color);
  }
  // Draw the hidden layer.
  for (int i = 0; i < NUM_HIDDEN; i++) {
    // Use the activation value to determine the color.
    Color color = weight_to_color(activations[NUM_INPUTS + i]);
    DrawCircle(300, 100 + i * 100, 30, color);
  }
  // Draw the output layer.
  for (int i = 0; i < NUM_OUTPUTS; i++) {
    // Use the activation value to determine the color.
    Color color = weight_to_color(activations[NUM_INPUTS + NUM_HIDDEN + i]);
    DrawCircle(500, 100 + i * 100, 30, color);
  }
  // Draw the weights.
  for (int i = 0; i < NUM_INPUTS; i++) {
    for (int j = 0; j < NUM_HIDDEN; j++) {
      // Use the weight value to determine the color.
      Color color = weight_to_color(hidden_weights[j][i]);
      DrawLineEx((Vector2){100 + 30, 100 + i * 100}, (Vector2){300 - 30, 100 + j * 100}, 2.0, color);
    }
  }
  for (int i = 0; i < NUM_HIDDEN; i++) {
    for (int j = 0; j < NUM_OUTPUTS; j++) {
      // Use the weight value to determine the color.
      Color color = weight_to_color(output_weights[i][j]);
      DrawLineEx((Vector2){300 + 30, 100 + i * 100}, (Vector2){500 - 30, 100 + j * 100}, 2.0, color);
    }
  }

  // Draw the labels.
  DrawText("Input", 100 - 30, 100 - 30, 20, BLACK);
  DrawText("Hidden", 300 - 30, 100 - 30, 20, BLACK);
  DrawText("Output", 500 - 30, 100 - 30, 20, BLACK);
}

int calculate_loss(int p) 
{
  double loss = 0.0;
  for (int i = 0; i < NUM_OUTPUTS; i++) {
    loss += pow(outputs[p][i] - output[i], 2);
  }
  return loss;
}

/**
 * Train the network.
 */
void *train_data(void *arg) 
{
  (void) arg;  // Cast arg to void to suppress the warning
  double best_validation_loss = DBL_MAX;
  int patience_counter = 0;
  if (running) {
    for (int e = epoch; e < NUM_EPOCHS; e++) {
      for (int p = 0; p < NUM_PATTERNS; p++) {
        forward_propagate(p);
        back_propagate(p);
      }

      usleep(DELAY);

      // TraceLog(LOG_INFO, "Epoch: %d / %d", e + 1, NUM_EPOCHS);

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
  }
  return NULL;
}

int main(void) {
  initialize_weights();
  initialize_biases();

  // Draw the network.
  InitWindow(WINDOW_WIDTH, WINDOW_HEIGHT, "Neural Network");
  SetTargetFPS(60);

  pthread_t thread;
  pthread_create(&thread, NULL, train_data, NULL);

  while (!WindowShouldClose()) {
    BeginDrawing();
    ClearBackground(RAYWHITE);

    double activations[NUM_INPUTS + NUM_HIDDEN + NUM_OUTPUTS];
    for (int i = 0; i < NUM_INPUTS; i++) {
      activations[i] = 0.0;
      for (int j = 0; j < NUM_HIDDEN; j++) {
        activations[i] += hidden_weights[i][j] * hidden[j];
      }
    }
    for (int i = 0; i < NUM_HIDDEN; i++) {
      activations[NUM_INPUTS + i] = hidden[i];
    }
    for (int i = 0; i < NUM_OUTPUTS; i++) {
      activations[NUM_INPUTS + NUM_HIDDEN + i] = output[i];
    }
    draw_network(activations);

    DrawText(TextFormat("Epoch: %d / %d", epoch + 1, NUM_EPOCHS), 10, 10, 20, BLACK);

    // Draw the input and outputs at the bottom of the screen.
    for (int p = 0; p < NUM_PATTERNS; p++) {
      forward_propagate(p);
      DrawText(TextFormat("Input: %d %d Output: %lf", (int)inputs[p][0], (int)inputs[p][1], output[0]), 10, 300 + p * 20, 20, BLACK);
    }

    EndDrawing();
  }

  // Close the thread when the window closes.
  running = 0;
  pthread_join(thread, NULL);
  CloseWindow();
  return 0;
}
