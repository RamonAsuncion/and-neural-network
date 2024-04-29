#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <float.h>
#include <pthread.h>
#include <unistd.h>

#include "config.h"
#include "raylib.h"

double inputs[NUM_PATTERNS][NUM_INPUTS] = {
  {0, 0},
  {0, 1},
  {1, 0},
  {1, 1}
};

double outputs[NUM_PATTERNS][NUM_OUTPUTS] = {
  {0},
  {0},
  {0},
  {1}
};

double hidden[NUM_HIDDEN];
double hidden_weights[NUM_INPUTS][NUM_HIDDEN];
double output_weights[NUM_HIDDEN][NUM_OUTPUTS];
double hidden_biases[NUM_HIDDEN];
double output_bias[NUM_OUTPUTS];
double output[NUM_OUTPUTS];

volatile int running = 1;
int epoch = 0;

double sigmoid(double x)
{
  return 1 / (1 + exp(-x));
}

double sigmoid_derivative(double x)
{
  return x * (1 - x);
}

void initialize_weights(void) {
  for (int i = 0; i < NUM_INPUTS; ++i) {
    for (int j = 0; j < NUM_HIDDEN; ++j) {
      hidden_weights[i][j] = (double) rand() / (double) RAND_MAX;
    }
  }
  for (int i = 0; i < NUM_HIDDEN; ++i) {
    for (int j = 0; j < NUM_OUTPUTS; ++j) {
      output_weights[i][j] = (double) rand() / (double) RAND_MAX;
    }
  }
}

void initialize_biases(void)
{
  for (int i = 0; i < NUM_HIDDEN; ++i) {
    hidden_biases[i] = (double) rand() / (double) RAND_MAX;
  }
  for (int i = 0; i < NUM_OUTPUTS; ++i) {
    output_bias[i] = (double) rand() / (double) RAND_MAX;
  }
}

void forward_propagate(int p)
{
  for (int i = 0; i < NUM_HIDDEN; ++i) {
    hidden[i] = 0.0;
    for (int j = 0; j < NUM_INPUTS; ++j) {
      hidden[i] += inputs[p][j] * hidden_weights[j][i];
    }
    hidden[i] += hidden_biases[i];
    hidden[i] = sigmoid(hidden[i]);
  }
  for (int i = 0; i < NUM_OUTPUTS; ++i) {
    output[i] = 0.0;
    for (int j = 0; j < NUM_HIDDEN; ++j) {
      output[i] += hidden[j] * output_weights[j][i];
    }
    output[i] += output_bias[i];
    output[i] = sigmoid(output[i]);
  }
}

void back_propagate(int p)
{
  double error;
  for (int i = 0; i < NUM_OUTPUTS; ++i) {
    error = outputs[p][i] - output[i];
    for (int j = 0; j < NUM_HIDDEN; ++j) {
      output_weights[j][i] += LEARNING_RATE * error * hidden[j] * sigmoid_derivative(output[i]);
    }
    output_bias[i] += error * sigmoid_derivative(output[i]);
  }
  for (int i = 0; i < NUM_HIDDEN; ++i) {
    error = 0.0;
    for (int j = 0; j < NUM_OUTPUTS; ++j) {
      error += output_weights[i][j] * (outputs[p][j] - output[j]);
    }
    for (int j = 0; j < NUM_INPUTS; ++j) {
      hidden_weights[j][i] += error * inputs[p][j] * sigmoid_derivative(hidden[i]);
    }
    hidden_biases[i] += LEARNING_RATE * error * sigmoid_derivative(hidden[i]);
  }
}

Color weight_to_color(double weight)
{
  double value = (weight + 1.0) / 2.0;
  int r = (int)(255 * value);
  int g = (int)(255 * (1 - value));
  int b = 0;
  return (Color){r, g, b, 255};
}

int is_neuron_used(int i)
{
  for (int j = 0; j < NUM_HIDDEN; ++j) {
    if (hidden_weights[i][j] != 0) {
      return 1;
    }
  }
  return 0;
}

int is_weight_used(int i, int j)
{
  if (hidden_weights[i][j] != 0) {
    return 1;
  }
  return 0;
}

void draw_network(double *activations)
{
  for (int i = 0; i < NUM_INPUTS; ++i) {
    Color color = weight_to_color(activations[i]);
    DrawCircle(100, 100 + i * 100, 30, color);
  }
  for (int i = 0; i < NUM_HIDDEN; ++i) {
    Color color = weight_to_color(activations[NUM_INPUTS + i]);
    DrawCircle(300, 100 + i * 100, 30, color);
  }
  for (int i = 0; i < NUM_OUTPUTS; ++i) {
    Color color = weight_to_color(activations[NUM_INPUTS + NUM_HIDDEN + i]);
    DrawCircle(500, 100 + i * 100, 30, color);
  }
  for (int i = 0; i < NUM_INPUTS; ++i) {
    for (int j = 0; j < NUM_HIDDEN; ++j) {
      Color color = weight_to_color(hidden_weights[j][i]);
      DrawLineEx((Vector2){100 + 30, 100 + i * 100}, (Vector2){300 - 30, 100 + j * 100}, 2.0, color);
    }
  }
  for (int i = 0; i < NUM_HIDDEN; ++i) {
    for (int j = 0; j < NUM_OUTPUTS; ++j) {
      Color color = weight_to_color(output_weights[i][j]);
      DrawLineEx((Vector2){300 + 30, 100 + i * 100}, (Vector2){500 - 30, 100 + j * 100}, 2.0, color);
    }
  }

  DrawText("Input", 100 - 30, 100 - 30, 20, BLACK);
  DrawText("Hidden", 300 - 30, 100 - 30, 20, BLACK);
  DrawText("Output", 500 - 30, 100 - 30, 20, BLACK);
}

int calculate_loss(int p)
{
  double loss = 0.0;
  for (int i = 0; i < NUM_OUTPUTS; ++i) {
    loss += pow(outputs[p][i] - output[i], 2);
  }
  return loss;
}

void *train_data(void *arg)
{
  (void) arg;
  double best_validation_loss = DBL_MAX;
  int patience_counter = 0;
  if (running) {
    for (int e = epoch; e < NUM_EPOCHS; e++) {
      for (int p = 0; p < NUM_PATTERNS; ++p) {
        forward_propagate(p);
        back_propagate(p);
      }

      sleep(RENDER_DELAY/1e6);

      double validation_loss = 0.0;
      for (int p = 0; p < NUM_PATTERNS; ++p) {
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

  InitWindow(WINDOW_WIDTH, WINDOW_HEIGHT, "Neural Network");
  SetTargetFPS(60);

  pthread_t thread;
  pthread_create(&thread, NULL, train_data, NULL);

  while (!WindowShouldClose()) {
    BeginDrawing();
    ClearBackground(RAYWHITE);

    double activations[NUM_INPUTS + NUM_HIDDEN + NUM_OUTPUTS];
    for (int i = 0; i < NUM_INPUTS; ++i) {
      activations[i] = 0.0;
      for (int j = 0; j < NUM_HIDDEN; ++j) {
        activations[i] += hidden_weights[i][j] * hidden[j];
      }
    }
    for (int i = 0; i < NUM_HIDDEN; ++i) {
      activations[NUM_INPUTS + i] = hidden[i];
    }
    for (int i = 0; i < NUM_OUTPUTS; ++i) {
      activations[NUM_INPUTS + NUM_HIDDEN + i] = output[i];
    }
    draw_network(activations);

    DrawText(TextFormat("Epoch: %d / %d", epoch + 1, NUM_EPOCHS), 10, 10, 20, BLACK);

    for (int p = 0; p < NUM_PATTERNS; ++p) {
      forward_propagate(p);
      DrawText(TextFormat("Input: %d %d Output: %lf", (int)inputs[p][0], (int)inputs[p][1], output[0]), 10, 300 + p * 20, 20, BLACK);
    }

    EndDrawing();
  }

  running = 0;
  pthread_join(thread, NULL);
  CloseWindow();
  return 0;
}

