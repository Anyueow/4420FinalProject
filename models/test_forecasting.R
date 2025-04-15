# Generic Forecasting Test Script

# Load required libraries
library(keras)
library(tensorflow)
library(readr)
library(dplyr)
library(tidyr)
library(reticulate)
library(ggplot2)

# Import Python modules
keras <- reticulate::import("tensorflow.keras")
np <- reticulate::import("numpy")

# Function to run forecasting test for a given category
run_forecasting_test <- function(category_type) {
  # -------------------------
  # 1. Data Loading & Preparation
  # -------------------------
  # Load the frequencies data
  data_file <- paste0("data/processed/", category_type, "_frequencies.csv")
  data <- read_csv(data_file)
  
  # Get unique seasons in order
  seasons <- unique(data$season)
  print(paste("Available seasons for", category_type, ":"))
  print(seasons)
  
  # Convert to wide format
  wide_data <- data %>%
    select(season, !!sym(category_type), percentage) %>%
    pivot_wider(
      names_from = season,
      values_from = percentage,
      id_cols = !!sym(category_type),
      values_fill = 0  # Fill missing values with 0
    )
  
  # Print the structure of the data
  print(paste("Data structure for", category_type, "after conversion:"))
  print(str(wide_data))
  print("First few rows:")
  print(head(wide_data))
  
  # Split data into training and test sets
  train_seasons <- c("Fall23", "Spring24", "Fall24")
  test_seasons <- c("Spring25", "Fall25")
  
  # Training data
  X_train <- as.matrix(wide_data[, train_seasons])
  y_train <- as.matrix(wide_data[, "Spring25"])  # Predict Spring25 from training data
  
  # Test data
  X_test <- as.matrix(wide_data[, c("Spring24", "Fall24", "Spring25")])
  y_test <- as.matrix(wide_data[, "Fall25"])  # Actual Fall25 data
  
  # Normalize the training data
  X_train_mean <- apply(X_train, 2, mean)
  X_train_sd <- apply(X_train, 2, sd)
  X_train_scaled <- scale(X_train)
  
  y_train_mean <- mean(y_train)
  y_train_sd <- sd(y_train)
  y_train_scaled <- scale(y_train)
  
  # Normalize test data using training statistics
  X_test_scaled <- scale(X_test, center = X_train_mean, scale = X_train_sd)
  
  # Reshape the input data to 3D array
  X_train_reshaped <- array(X_train_scaled, dim = c(nrow(X_train_scaled), ncol(X_train_scaled), 1))
  X_test_reshaped <- array(X_test_scaled, dim = c(nrow(X_test_scaled), ncol(X_test_scaled), 1))
  
  # -------------------------
  # 2. Model Building
  # -------------------------
  timesteps <- dim(X_train_reshaped)[2]
  num_features <- dim(X_train_reshaped)[3]
  
  # Convert R arrays to numpy arrays
  X_train_np <- np$array(X_train_reshaped)
  y_train_np <- np$array(y_train_scaled)
  X_test_np <- np$array(X_test_reshaped)
  
  # Create model
  model <- keras$Sequential()
  model$add(keras$layers$LSTM(units = 16L, input_shape = list(timesteps, num_features)))
  model$add(keras$layers$Dense(units = 1L))
  
  # Compile model
  model$compile(
    loss = 'mean_squared_error',
    optimizer = keras$optimizers$Adam(learning_rate = 0.001)
  )
  
  # -------------------------
  # 3. Model Training and Evaluation
  # -------------------------
  # Train the model
  history <- model$fit(
    X_train_np, y_train_np,
    epochs = 50L,
    batch_size = 4L,
    validation_split = 0.2,
    verbose = 1L
  )
  
  # Make predictions
  predictions_scaled <- model$predict(X_test_np)
  predictions <- as.numeric(predictions_scaled) * y_train_sd + y_train_mean
  
  # -------------------------
  # 4. Results Analysis
  # -------------------------
  # Create results data frame
  results_df <- data.frame(
    category = wide_data[[category_type]],
    actual_fall25 = as.numeric(y_test),
    predicted_fall25 = as.numeric(predictions),
    error = abs(as.numeric(y_test) - as.numeric(predictions))
  )
  
  # Calculate metrics
  mae <- mean(results_df$error, na.rm = TRUE)
  rmse <- sqrt(mean(results_df$error^2, na.rm = TRUE))
  
  print(paste("\nModel Performance Metrics for", category_type, ":"))
  print(paste("Mean Absolute Error (MAE):", round(mae, 2)))
  print(paste("Root Mean Square Error (RMSE):", round(rmse, 2)))
  
  # Sort by actual value and add confidence levels
  results_df <- results_df %>%
    arrange(desc(actual_fall25)) %>%
    mutate(
      confidence = case_when(
        abs(scale(predicted_fall25)) > 1.5 ~ "High",
        abs(scale(predicted_fall25)) < 0.5 ~ "Low",
        TRUE ~ "Medium"
      )
    )
  
  print(paste("\nTop", category_type, "by actual Fall25 frequency:"))
  print(results_df)
  
  # Create directory if it doesn't exist
  if (!dir.exists("data/lstm")) {
    dir.create("data/lstm", recursive = TRUE)
  }
  
  # Save results
  output_file <- paste0("data/lstm/", category_type, "_forecasting_test_results.csv")
  write_csv(results_df, output_file)
  
  # -------------------------
  # 5. Visualization
  # -------------------------
  # Create a scatter plot of actual vs predicted values
  p <- ggplot(results_df, aes(x = actual_fall25, y = predicted_fall25)) +
    geom_point(aes(color = confidence)) +
    geom_abline(intercept = 0, slope = 1, linetype = "dashed") +
    labs(
      title = paste("Actual vs Predicted", category_type, "Frequencies for Fall25"),
      x = "Actual Frequency (%)",
      y = "Predicted Frequency (%)",
      color = "Confidence Level"
    ) +
    theme_minimal()
  
  # Save the plot
  plot_file <- paste0("data/lstm/", category_type, "_forecasting_test_plot.png")
  ggsave(plot_file, p, width = 10, height = 6)
  
  return(results_df)
}

# Run tests for all categories except color
categories <- c("category", "pattern", "super_category", "style")

# Create a list to store results
results <- list()

# Run tests for each category
for (cat in categories) {
  print(paste("\nRunning test for", cat))
  results[[cat]] <- run_forecasting_test(cat)
}

# Print summary of all tests
print("\nSummary of all tests:")
for (cat in categories) {
  mae <- mean(results[[cat]]$error, na.rm = TRUE)
  rmse <- sqrt(mean(results[[cat]]$error^2, na.rm = TRUE))
  print(paste(cat, "MAE:", round(mae, 2), "RMSE:", round(rmse, 2)))
} 