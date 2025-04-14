# In this R file there is 3 differnet sections the first was a testing of the 
# gtrends library and using specific clothing tags the second is more specific
# tags based on data gathered from another anlaysis we had the last is due to
# being blocked by gtrend we had to download a csv file so our code could be 
# graded hence we have two .rmd as well one with the csv the other wihtout 

library(gtrendsR)

# Example fashion search terms
keywords <- c("baggy jeans", "cargo pants", "leather jacket")

trends <- gtrends(
  keyword = keywords,
  time = "today+5-y",     # ⏳ 5-year range
  gprop = "web",          # or "images", "youtube" etc.
  geo = ""                # set to "US" or "GB" etc. to localize
)

write.csv(trends$interest_over_time, "google_fashion_trends.csv", row.names = FALSE)

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

plot(ts_data, main = "Google Fashion Trends Time Series")
decomposed <- decompose(ts_data[, "baggy jeans"], type = "multiplicative")
plot(decomposed)




library(gtrendsR)
library(dplyr)
library(ggplot2)
library(tidyr)
library(lubridate)

keywords2 <- c("turtleneck sweater", "corset-style belt", "shearling slippers", "shearling-lined clogs")
results2 <- list()

for (k in keywords2) {
  cat("Fetching:", k, "\n")
  result <- tryCatch(
    {
      Sys.sleep(20)  # Wait to avoid 429 error
      gtrends(k, time = "today+5-y", gprop = "web", geo = "")
    },
    error = function(e) {
      message("Error fetching keyword:", k)
      message(e)
      return(NULL)
    }
  )
  results2[[k]] <- result
}


df2_list <- lapply(names(results2), function(k) {
  if (!is.null(results2[[k]])) {
    df <- results2[[k]]$interest_over_time
    df$keyword <- k  # overwrite with consistent label
    return(df)
  }
  return(NULL)
})

df2_combined <- bind_rows(df2_list)
write.csv(df2_combined, "google_fashion_trends-2.csv", row.names = FALSE)

# Plot
df2_combined %>%
  ggplot(aes(x = date, y = hits, color = keyword)) +
  geom_line(linewidth = 1) +
  labs(title = "Fashion Term Popularity Over Time (Google Trends) – Group 2",
       x = "Date", y = "Search Interest") +
  theme_minimal()

# Time series prep
df2 <- df2_combined %>%
  select(date, keyword, hits) %>%
  spread(keyword, hits)

ts_data2 <- ts(
  df2[,-1],
  start = c(year(min(df2$date)), month(min(df2$date))),
  frequency = 12
)

plot(ts_data2, main = "Google Fashion Trends Time Series (Group 2)")

if ("turtleneck sweater" %in% colnames(ts_data2)) {
  decomposed2 <- decompose(ts_data2[, "turtleneck sweater"], type = "multiplicative")
  plot(decomposed2)
}

write.csv(df2, "google_fashion_trends_cleaned-2.csv", row.names = FALSE)


# Load libraries
library(readr)
library(dplyr)
library(tidyr)
library(ggplot2)
library(lubridate)

df <- read_csv("combined_fashion_trends_fixed.csv")
head(df)

head(df)

df_wide <- df %>%
  select(date, keyword, hits) %>%
  pivot_wider(names_from = keyword, values_from = hits)

df_wide$date <- as.Date(df_wide$date)

ts_data <- ts(df_wide[,-1],
              start = c(year(min(df_wide$date)), month(min(df_wide$date))),
              frequency = 52)  # Weekly data

plot(ts_data, main = "Google Trends Time Series for Fashion Keywords")

decomposed <- decompose(ts_data[, "turtleneck sweater"], type = "multiplicative")
plot(decomposed)

