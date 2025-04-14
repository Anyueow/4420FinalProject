# Required packages
library(tidyverse)
library(lubridate)
library(zoo)
library(scales)
library(caret)  # For data preprocessing

# Read and process fashion labels data
fashion_data <- read.csv("data/processed/fashion_labels.csv", stringsAsFactors = FALSE)

# Extract season from image path and create a proper time order
fashion_data$season <- str_extract(fashion_data$image_path, "Fall24|Spring25|Fall25")
fashion_data$season_order <- case_when(
  fashion_data$season == "Fall24" ~ 1,
  fashion_data$season == "Spring25" ~ 2,
  fashion_data$season == "Fall25" ~ 3
)

# Add a proper chronological factor for plotting
fashion_data$season <- factor(fashion_data$season, 
                            levels = c("Fall24", "Spring25", "Fall25"),
                            ordered = TRUE)

# Function to calculate comprehensive feature frequencies
calculate_feature_frequencies <- function(data) {
  # Style frequencies
  style_freq <- data %>%
    filter(style != "", !is.na(style)) %>%
    group_by(season, season_order, style) %>%
    summarise(frequency = n(), .groups = "drop") %>%
    group_by(season) %>%
    mutate(percentage = (frequency / sum(frequency)) * 100) %>%
    ungroup()
  
  # Category frequencies
  category_freq <- data %>%
    filter(category != "", !is.na(category)) %>%
    group_by(season, season_order, category) %>%
    summarise(frequency = n(), .groups = "drop") %>%
    group_by(season) %>%
    mutate(percentage = (frequency / sum(frequency)) * 100) %>%
    ungroup()
  
  # Super category frequencies
  super_category_freq <- data %>%
    filter(super_category != "", !is.na(super_category)) %>%
    group_by(season, season_order, super_category) %>%
    summarise(frequency = n(), .groups = "drop") %>%
    group_by(season) %>%
    mutate(percentage = (frequency / sum(frequency)) * 100) %>%
    ungroup()
  
  # Pattern frequencies
  pattern_freq <- data %>%
    filter(pattern != "", !is.na(pattern)) %>%
    group_by(season, season_order, pattern) %>%
    summarise(frequency = n(), .groups = "drop") %>%
    group_by(season) %>%
    mutate(percentage = (frequency / sum(frequency)) * 100) %>%
    ungroup()
  
  # Ensure chronological order in all dataframes
  style_freq$season <- factor(style_freq$season, 
                            levels = c("Fall24", "Spring25", "Fall25"),
                            ordered = TRUE)
  category_freq$season <- factor(category_freq$season, 
                               levels = c("Fall24", "Spring25", "Fall25"),
                               ordered = TRUE)
  super_category_freq$season <- factor(super_category_freq$season, 
                                     levels = c("Fall24", "Spring25", "Fall25"),
                                     ordered = TRUE)
  pattern_freq$season <- factor(pattern_freq$season, 
                              levels = c("Fall24", "Spring25", "Fall25"),
                              ordered = TRUE)
  
  list(
    style = style_freq,
    category = category_freq,
    super_category = super_category_freq,
    pattern = pattern_freq
  )
}

# Calculate all feature frequencies
feature_frequencies <- calculate_feature_frequencies(fashion_data)

# Save feature frequencies
write.csv(feature_frequencies$style, "data/processed/style_frequencies.csv", row.names = FALSE)
write.csv(feature_frequencies$category, "data/processed/category_frequencies.csv", row.names = FALSE)
write.csv(feature_frequencies$super_category, "data/processed/super_category_frequencies.csv", row.names = FALSE)
write.csv(feature_frequencies$pattern, "data/processed/pattern_frequencies.csv", row.names = FALSE)

# Function to calculate trend momentum
calculate_trend_momentum <- function(freq_data, feature_col) {
  freq_data %>%
    arrange(season_order) %>%
    group_by(!!sym(feature_col)) %>%
    mutate(
      prev_percentage = lag(percentage),
      momentum = (percentage - prev_percentage) / prev_percentage * 100,
      avg_momentum = mean(momentum, na.rm = TRUE)
    ) %>%
    ungroup()
}

# Calculate trend momentum for all features
style_momentum <- calculate_trend_momentum(feature_frequencies$style, "style")
category_momentum <- calculate_trend_momentum(feature_frequencies$category, "category")
super_category_momentum <- calculate_trend_momentum(feature_frequencies$super_category, "super_category")
pattern_momentum <- calculate_trend_momentum(feature_frequencies$pattern, "pattern")

# Function to predict next season using weighted average and momentum
predict_next_season <- function(freq_data, momentum_data, feature_col) {
  # Get the most recent season's data (Fall25)
  latest_season <- "Fall25"
  
  # Calculate weighted average considering momentum
  predictions <- freq_data %>%
    filter(season == latest_season) %>%
    left_join(
      momentum_data %>%
        select(!!sym(feature_col), avg_momentum),
      by = feature_col
    ) %>%
    mutate(
      predicted = percentage * (1 + coalesce(avg_momentum, 0) / 100),
      confidence = case_when(
        !is.na(avg_momentum) & avg_momentum > 0 ~ "High",
        !is.na(avg_momentum) & avg_momentum < 0 ~ "Low",
        TRUE ~ "Medium"
      )
    ) %>%
    select(!!sym(feature_col), predicted, confidence) %>%
    arrange(desc(predicted))
  
  predictions
}

# Make predictions for next season
style_predictions <- predict_next_season(feature_frequencies$style, style_momentum, "style")
category_predictions <- predict_next_season(feature_frequencies$category, category_momentum, "category")
super_category_predictions <- predict_next_season(feature_frequencies$super_category, super_category_momentum, "super_category")
pattern_predictions <- predict_next_season(feature_frequencies$pattern, pattern_momentum, "pattern")

# Save predictions
write.csv(style_predictions, "data/processed/style_predictions.csv", row.names = FALSE)
write.csv(category_predictions, "data/processed/category_predictions.csv", row.names = FALSE)
write.csv(super_category_predictions, "data/processed/super_category_predictions.csv", row.names = FALSE)
write.csv(pattern_predictions, "data/processed/pattern_predictions.csv", row.names = FALSE)

# Function to calculate trend statistics
calculate_trend_stats <- function(freq_data, feature_col) {
  freq_data %>%
    group_by(!!sym(feature_col)) %>%
    summarise(
      avg_percentage = mean(percentage),
      trend_direction = case_when(
        n() >= 2 & last(percentage) > first(percentage) ~ "Increasing",
        n() >= 2 & last(percentage) < first(percentage) ~ "Decreasing",
        TRUE ~ "Stable"
      ),
      volatility = sd(percentage) / mean(percentage) * 100
    ) %>%
    arrange(desc(avg_percentage))
}

# Calculate trend statistics for all features
style_trend_stats <- calculate_trend_stats(feature_frequencies$style, "style")
category_trend_stats <- calculate_trend_stats(feature_frequencies$category, "category")
super_category_trend_stats <- calculate_trend_stats(feature_frequencies$super_category, "super_category")
pattern_trend_stats <- calculate_trend_stats(feature_frequencies$pattern, "pattern")

# Save trend statistics
write.csv(style_trend_stats, "data/processed/style_trend_stats.csv", row.names = FALSE)
write.csv(category_trend_stats, "data/processed/category_trend_stats.csv", row.names = FALSE)
write.csv(super_category_trend_stats, "data/processed/super_category_trend_stats.csv", row.names = FALSE)
write.csv(pattern_trend_stats, "data/processed/pattern_trend_stats.csv", row.names = FALSE)

# Print predictions with confidence levels
cat("\nPredicted Top Style Trends for Spring26:\n")
style_predictions %>%
  top_n(5, predicted) %>%
  mutate(
    predicted = round(predicted, 2),
    confidence = factor(confidence, levels = c("High", "Medium", "Low"))
  ) %>%
  print(n = 5)

cat("\nPredicted Top Category Trends for Spring26:\n")
category_predictions %>%
  top_n(5, predicted) %>%
  mutate(
    predicted = round(predicted, 2),
    confidence = factor(confidence, levels = c("High", "Medium", "Low"))
  ) %>%
  print(n = 5)

cat("\nPredicted Top Super Category Trends for Spring26:\n")
super_category_predictions %>%
  top_n(5, predicted) %>%
  mutate(
    predicted = round(predicted, 2),
    confidence = factor(confidence, levels = c("High", "Medium", "Low"))
  ) %>%
  print(n = 5)

cat("\nPredicted Top Pattern Trends for Spring26:\n")
pattern_predictions %>%
  top_n(5, predicted) %>%
  mutate(
    predicted = round(predicted, 2),
    confidence = factor(confidence, levels = c("High", "Medium", "Low"))
  ) %>%
  print(n = 5)

# Print trend statistics
cat("\nStyle Trend Statistics:\n")
style_trend_stats %>%
  top_n(5, avg_percentage) %>%
  print(n = 5)

cat("\nCategory Trend Statistics:\n")
category_trend_stats %>%
  top_n(5, avg_percentage) %>%
  print(n = 5)

cat("\nSuper Category Trend Statistics:\n")
super_category_trend_stats %>%
  top_n(5, avg_percentage) %>%
  print(n = 5)

cat("\nPattern Trend Statistics:\n")
pattern_trend_stats %>%
  top_n(5, avg_percentage) %>%
  print(n = 5)

# Save trend summary
trend_summary <- list(
  style_trends = list(
    top_trends = style_predictions %>% top_n(5, predicted),
    trend_statistics = style_trend_stats %>% top_n(5, avg_percentage),
    momentum_analysis = style_momentum %>% 
      filter(!is.na(momentum)) %>%
      summarise(
        increasing_trends = sum(momentum > 0, na.rm = TRUE),
        decreasing_trends = sum(momentum < 0, na.rm = TRUE),
        stable_trends = sum(momentum == 0, na.rm = TRUE)
      )
  ),
  category_trends = list(
    top_trends = category_predictions %>% top_n(5, predicted),
    trend_statistics = category_trend_stats %>% top_n(5, avg_percentage),
    momentum_analysis = category_momentum %>% 
      filter(!is.na(momentum)) %>%
      summarise(
        increasing_trends = sum(momentum > 0, na.rm = TRUE),
        decreasing_trends = sum(momentum < 0, na.rm = TRUE),
        stable_trends = sum(momentum == 0, na.rm = TRUE)
      )
  ),
  super_category_trends = list(
    top_trends = super_category_predictions %>% top_n(5, predicted),
    trend_statistics = super_category_trend_stats %>% top_n(5, avg_percentage),
    momentum_analysis = super_category_momentum %>% 
      filter(!is.na(momentum)) %>%
      summarise(
        increasing_trends = sum(momentum > 0, na.rm = TRUE),
        decreasing_trends = sum(momentum < 0, na.rm = TRUE),
        stable_trends = sum(momentum == 0, na.rm = TRUE)
      )
  ),
  pattern_trends = list(
    top_trends = pattern_predictions %>% top_n(5, predicted),
    trend_statistics = pattern_trend_stats %>% top_n(5, avg_percentage),
    momentum_analysis = pattern_momentum %>% 
      filter(!is.na(momentum)) %>%
      summarise(
        increasing_trends = sum(momentum > 0, na.rm = TRUE),
        decreasing_trends = sum(momentum < 0, na.rm = TRUE),
        stable_trends = sum(momentum == 0, na.rm = TRUE)
      )
  )
)

# Save trend summary to file
writeLines(
  c(
    "Top Predicted Style Trends for Spring26:",
    paste(style_predictions$style[1:5], ":", round(style_predictions$predicted[1:5], 2), "%"),
    "\nTop Predicted Category Trends for Spring26:",
    paste(category_predictions$category[1:5], ":", round(category_predictions$predicted[1:5], 2), "%"),
    "\nTop Predicted Super Category Trends for Spring26:",
    paste(super_category_predictions$super_category[1:5], ":", round(super_category_predictions$predicted[1:5], 2), "%"),
    "\nTop Predicted Pattern Trends for Spring26:",
    paste(pattern_predictions$pattern[1:5], ":", round(pattern_predictions$predicted[1:5], 2), "%"),
    "\nStyle Trend Statistics:",
    paste("Average Style Trend Volatility:", round(mean(style_trend_stats$volatility, na.rm = TRUE), 2), "%"),
    paste("Number of Increasing Style Trends:", trend_summary$style_trends$momentum_analysis$increasing_trends),
    paste("Number of Decreasing Style Trends:", trend_summary$style_trends$momentum_analysis$decreasing_trends),
    paste("Number of Stable Style Trends:", trend_summary$style_trends$momentum_analysis$stable_trends),
    "\nCategory Trend Statistics:",
    paste("Average Category Trend Volatility:", round(mean(category_trend_stats$volatility, na.rm = TRUE), 2), "%"),
    paste("Number of Increasing Category Trends:", trend_summary$category_trends$momentum_analysis$increasing_trends),
    paste("Number of Decreasing Category Trends:", trend_summary$category_trends$momentum_analysis$decreasing_trends),
    paste("Number of Stable Category Trends:", trend_summary$category_trends$momentum_analysis$stable_trends),
    "\nSuper Category Trend Statistics:",
    paste("Average Super Category Trend Volatility:", round(mean(super_category_trend_stats$volatility, na.rm = TRUE), 2), "%"),
    paste("Number of Increasing Super Category Trends:", trend_summary$super_category_trends$momentum_analysis$increasing_trends),
    paste("Number of Decreasing Super Category Trends:", trend_summary$super_category_trends$momentum_analysis$decreasing_trends),
    paste("Number of Stable Super Category Trends:", trend_summary$super_category_trends$momentum_analysis$stable_trends),
    "\nPattern Trend Statistics:",
    paste("Average Pattern Trend Volatility:", round(mean(pattern_trend_stats$volatility, na.rm = TRUE), 2), "%"),
    paste("Number of Increasing Pattern Trends:", trend_summary$pattern_trends$momentum_analysis$increasing_trends),
    paste("Number of Decreasing Pattern Trends:", trend_summary$pattern_trends$momentum_analysis$decreasing_trends),
    paste("Number of Stable Pattern Trends:", trend_summary$pattern_trends$momentum_analysis$stable_trends)
  ),
  "data/processed/trend_summary.txt"
) 