# Set CRAN mirror first
options(repos = c(CRAN = "https://cloud.r-project.org"))

# Function to install missing packages with error handling
install_if_missing <- function(package) {
    tryCatch({
        if (!requireNamespace(package, character.only = TRUE, quietly = TRUE)) {
            message(sprintf("\nInstalling package: %s", package))
            install.packages(package, dependencies = TRUE)
            if (!requireNamespace(package, character.only = TRUE, quietly = TRUE)) {
                stop(sprintf("Failed to install package: %s", package))
            }
        } else {
            message(sprintf("Package already installed: %s", package))
        }
    }, error = function(e) {
        message(sprintf("Error with package %s: %s", package, e$message))
    })
}

# Install required packages if not already installed
message("Installing required R packages...")
required_packages <- c(
    "tidyverse",
    "keras",
    "tensorflow",
    "reticulate",
    "caret",
    "forecast",
    "lubridate",
    "zoo",
    "scales",
    "futile.logger"
)

# Install packages one by one
for (package in required_packages) {
    install_if_missing(package)
}

# Initialize tensorflow with error handling
message("\nSetting up TensorFlow...")
tryCatch({
    if (!requireNamespace("tensorflow")) {
        install.packages("tensorflow")
    }
    library(tensorflow)
    
    # Check if tensorflow is installed in Python
    if (!tf_config()$available) {
        message("Installing TensorFlow via reticulate...")
        reticulate::py_install("tensorflow", pip = TRUE)
    }
    
    # Test tensorflow
    tf_version()
    message("TensorFlow setup complete!")
}, error = function(e) {
    message(sprintf("Error setting up TensorFlow: %s", e$message))
    message("Please install TensorFlow manually using: pip install tensorflow")
})

message("\nSetup complete! You can now run the ML forecasting script.") 