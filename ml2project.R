# R package
library(gtrendsR)

# Example fashion search terms
keywords <- c("baggy jeans", "cargo pants", "leather jacket")

# Get interest over time for the last 5 years
trends <- gtrends(
  keyword = keywords,
  time = "today+5-y",     # ⏳ 5-year range
  gprop = "web",          # or "images", "youtube" etc.
  geo = ""                # set to "US" or "GB" etc. to localize
)
# Save raw trends data to CSV
write.csv(trends$interest_over_time, "google_fashion_trends.csv", row.names = FALSE)

# Preview
head(trends$interest_over_time)

library(ggplot2)
library(dplyr)

trends$interest_over_time %>%
  ggplot(aes(x = date, y = hits, color = keyword)) +
  geom_line(size = 1) +
  labs(title = "Fashion Term Popularity Over Time (Google Trends)",
       x = "Date", y = "Search Interest") +
  theme_minimal()

library(tidyr)

df <- trends$interest_over_time %>%
  select(date, keyword, hits) %>%
  spread(keyword, hits)
library(lubridate)
# Convert to ts object
ts_data <- ts(df[,-1], start = c(year(min(df$date)), month(min(df$date))), frequency = 12)

# Plot all
plot(ts_data, main = "Google Fashion Trends Time Series")
decomposed <- decompose(ts_data[, "baggy jeans"], type = "multiplicative")
plot(decomposed)
