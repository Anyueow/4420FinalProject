# Trend Prediction Script

library(keras)
library(tensorflow)
library(readr)
library(dplyr)
library(tidyr)
library(reticulate)

keras <- reticulate::import("tensorflow.keras")
np <- reticulate::import("numpy")

# Function to predict trends for a given category
predict_trends <- function(category_type) {

  # find all the frequencies data generated from actual_trends.R
  data_file <- paste0("data/processed/", category_type, "_frequencies.csv")
  data <- read_csv(data_file)
  
  # Get unique seasons in order and exclude Fall'25
  seasons <- unique(data$season)
  seasons <- seasons[seasons != "Fall'25"]
  print(paste("Available seasons for", category_type, ":"))
  print(seasons)
  
  # Convert to wide format
  wide_data <- data %>%
    filter(season != "Fall'25") %>%  # Exclude Fall'25
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
  
  # Use all available seasons for training
  train_seasons <- colnames(wide_data)[-1]  # Exclude the category column
  
  # Training data
  X <- as.matrix(wide_data[, train_seasons])
  
  # Normalize the data
  X_mean <- apply(X, 2, mean)
  X_sd <- apply(X, 2, sd)
  X_scaled <- scale(X)
  
  # Reshape the input data to 3D array
  X_reshaped <- array(X_scaled, dim = c(nrow(X_scaled), ncol(X_scaled), 1))
  

  timesteps <- dim(X_reshaped)[2]
  num_features <- dim(X_reshaped)[3]
  
  # Convert R arrays to numpy arrays with proper shape and type
  X_np <- np$array(X_reshaped, dtype = "float32")
  
  # Create model
  model <- keras$Sequential()
  model$add(keras$layers$Input(shape = list(timesteps, num_features)))
  model$add(keras$layers$LSTM(units = 16L))
  model$add(keras$layers$Dense(units = 1L))
  
  # Compile model
  model$compile(
    loss = 'mean_squared_error',
    optimizer = keras$optimizers$Adam(learning_rate = 0.001)
  )
  
  
  X_train <- X_np[, -ncol(X_np), , drop = FALSE]  # All seasons except last
  y_train <- X_np[, ncol(X_np), 1]  # Last season as target
  
  # Split data into training and validation sets
  train_size <- floor(0.8 * nrow(X_train))
  indices <- sample(seq_len(nrow(X_train)))
  train_indices <- indices[1:train_size]
  val_indices <- indices[(train_size + 1):length(indices)]
  
  X_train_split <- X_train[train_indices, , , drop = FALSE]
  y_train_split <- y_train[train_indices]
  X_val <- X_train[val_indices, , , drop = FALSE]
  y_val <- y_train[val_indices]
  
  # Ensure all arrays are float32
  X_train_split <- np$array(X_train_split, dtype = "float32")
  y_train_split <- np$array(y_train_split, dtype = "float32")
  X_val <- np$array(X_val, dtype = "float32")
  y_val <- np$array(y_val, dtype = "float32")
  
  # Train the model
  history <- model$fit(
    X_train_split, y_train_split,
    epochs = 50L,
    batch_size = 4L,
    validation_data = list(X_val, y_val),
    verbose = 1L
  )
  
  #making predictions
  # Prepare input for prediction (last n-1 seasons)
  X_pred <- X_np[, -1, , drop = FALSE]  # Remove first season
  X_pred <- np$array(X_pred, dtype = "float32")  
  
  predictions_scaled <- model$predict(X_pred)
  predictions <- as.numeric(predictions_scaled) * X_sd[ncol(X)] + X_mean[ncol(X)]
  
  #creating a dataframe with the predicted values
  results_df <- data.frame(
    category = wide_data[[category_type]],
    predicted = predictions
  )
  
  # Sort by predicted value and add confidence levels
  results_df <- results_df %>%
    arrange(desc(predicted)) %>%
    mutate(
      confidence = case_when(
        abs(scale(predicted)) > 1.5 ~ "High",
        abs(scale(predicted)) < 0.5 ~ "Low",
        TRUE ~ "Medium"
      )
    )
  
  print(paste("\nTop predicted", category_type, ":"))
  print(head(results_df, 5))
  
  if (!dir.exists("data/predictions")) {
    dir.create("data/predictions", recursive = TRUE)
  }
  
  # Save results
  output_file <- paste0("data/predictions/", category_type, "_predictions.csv")
  write_csv(results_df, output_file)
  
  return(results_df)
}

# Run predictions for all categories
categories <- c("category", "pattern", "super_category", "style", "color")

# Create a list to store results
results <- list()

for (cat in categories) {
  print(paste("\nRunning predictions for", cat))
  results[[cat]] <- predict_trends(cat)
}

# Print summary of all predictions
print("\nSummary of all predictions:")
for (cat in categories) {
  top_5 <- head(results[[cat]], 5)
  print(paste("\nTop 5 predicted", cat, "for Fall'25:"))
  print(top_5)
} 