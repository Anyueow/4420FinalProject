# Minimal LSTM Forecasting on Color Combinations

# Load required libraries
library(keras)
library(tensorflow)
library(readr)
library(dplyr)
library(tidyr)
library(reticulate)

# Import Python modules
keras <- reticulate::import("tensorflow.keras")
np <- reticulate::import("numpy")

# -------------------------
# 1. Data Loading & Preparation
# -------------------------
# Load the color combination frequencies data
data <- read_csv("data/processed/color_combo_frequencies.csv")

# Get unique seasons in order
seasons <- unique(data$season)
print("Available seasons:")
print(seasons)

# Convert to wide format
wide_data <- data %>%
  select(season, color_1, color_2, percentage) %>%
  unite("color_combo", color_1, color_2, sep = " + ") %>%
  pivot_wider(
    names_from = season,
    values_from = percentage,
    id_cols = color_combo,
    values_fill = 0  # Fill missing values with 0
  )

# Print the structure of the data
print("Data structure after conversion:")
print(str(wide_data))
print("First few rows:")
print(head(wide_data))

# Extract color combinations and the season values
color_combos <- wide_data$color_combo

# Use all seasons except the last one for training
input_seasons <- head(seasons, -1)
target_season <- tail(seasons, 1)

print(paste("Using seasons for training:", paste(input_seasons, collapse=", ")))
print(paste("Predicting for season:", target_season))

# Use available seasons as input features
X <- as.matrix(wide_data[, input_seasons])
# The target is the last available season
y <- as.matrix(wide_data[, target_season])

# Print dimensions for debugging
print("Input dimensions:")
print(dim(X))
print("Target dimensions:")
print(dim(y))

# Normalize the input data (z-score normalization, per column)
X_mean <- apply(X, 2, mean)
X_sd <- apply(X, 2, sd)
X_scaled <- scale(X)

# Normalize the target
y_mean <- mean(y)
y_sd <- sd(y)
y_scaled <- scale(y)

# Reshape the input data to 3D array: (samples, timesteps, features)
# Each sample is a sequence of timesteps (one per season) with one feature per timestep.
X_reshaped <- array(X_scaled, dim = c(nrow(X_scaled), ncol(X_scaled), 1))

print("Reshaped input dimensions:")
print(dim(X_reshaped))

# -------------------------
# 2. Model Building
# -------------------------
timesteps <- dim(X_reshaped)[2]    # Number of seasons used for prediction
num_features <- dim(X_reshaped)[3] # Expected to be 1

print(paste("Timesteps:", timesteps))
print(paste("Features:", num_features))

# Convert R arrays to numpy arrays
X_np <- np$array(X_reshaped)
y_np <- np$array(y_scaled)

# Create model using Python syntax
model <- keras$Sequential()
model$add(keras$layers$LSTM(units = 16L, input_shape = list(timesteps, num_features)))
model$add(keras$layers$Dense(units = 1L))

# Compile model
model$compile(
  loss = 'mean_squared_error',
  optimizer = keras$optimizers$Adam(learning_rate = 0.001)
)

# Print model summary
model$summary()

# -------------------------
# 3. Model Training and Prediction
# -------------------------
# Train the model
history <- model$fit(
  X_np, y_np,
  epochs = 50L,
  batch_size = 4L,
  validation_split = 0.2,
  verbose = 1L
)

# Make predictions
predictions_scaled <- model$predict(X_np)

# Convert predictions back to the original scale
predictions <- as.numeric(predictions_scaled) * y_sd + y_mean

# -------------------------
# 4. Output the Results
# -------------------------
# Create a data frame with color combinations and their predicted values
predicted_df <- data.frame(
  color_combo = color_combos,
  predicted = predictions
)

# Sort by predicted value and add confidence levels
predicted_df <- predicted_df %>%
  arrange(desc(predicted)) %>%
  mutate(
    confidence = case_when(
      abs(scale(predicted)) > 1.5 ~ "High",
      abs(scale(predicted)) < 0.5 ~ "Low",
      TRUE ~ "Medium"
    )
  )

print(paste("Top predicted color combinations for", target_season, ":"))
print(predicted_df)

# Create directory if it doesn't exist
if (!dir.exists("data/lstm")) {
  dir.create("data/lstm", recursive = TRUE)
}

# Save predictions
write_csv(predicted_df, "data/lstm/color_combo_lstm_predictions.csv")
