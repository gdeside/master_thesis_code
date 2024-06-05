# Load necessary libraries
library(lme4)  # For mixed models
library(emmeans)  # For post-hoc tests
library(reshape2)  # For reshaping the data
library(ggplot2)  # For plotting
library(dplyr)  # For data manipulation
library(stringr)  # For string splitting

# Read the data
data <- read.csv("/Users/guillaumedeside/Downloads/DTI_AD_values_comparison_b1000.csv")  # Replace with the actual file path

# Check for missing values
if (any(is.na(data))) {
  stop("Missing values detected. Please handle missing values before proceeding.")
}

# Convert 'Patient ID' to a factor
data$`Patient.ID` <- factor(data$`Patient.ID`)

# Reshape the data to long format
data_long <- melt(data, id.vars = "Patient.ID", 
                  variable.name = "reduced_variable", 
                  value.name = "value")

# Convert the 'value' column to numeric
data_long$value <- as.numeric(data_long$value)

# Convert 'reduced_variable' to a factor
data_long$reduced_variable <- factor(data_long$reduced_variable, levels = unique(data_long$reduced_variable))

# Define and fit the GLMM
model <- lmer(value ~ reduced_variable + (1 | `Patient.ID`), data = data_long)

# Conduct post-hoc comparisons using emmeans
posthoc_tests <- emmeans(model, pairwise ~ reduced_variable)

# Extract p-values from posthoc_tests
p_values <- summary(posthoc_tests$contrasts)

# Create pairs for annotation
pair_combinations <- combn(levels(data_long$reduced_variable), 2, simplify = TRUE)

# Create a dataframe to store the pairs and corresponding p-values
p_values <- data.frame(pair1 = pair_combinations[1, ], 
                       pair2 = pair_combinations[2, ], 
                       stringsAsFactors = FALSE)

# Add p-values to the dataframe
p_values$p.value <- apply(p_values, 1, function(pair) {
  contrast <- paste(pair, collapse = " vs ")
  summary(posthoc_tests$contrasts)[contrast, "p.value"]
})

# Create the boxplot
p <- ggplot(data_long, aes(x = reduced_variable, y = value, fill = reduced_variable)) +
  geom_boxplot(outlier.shape = NA) +  # Removes the outliers from the plot
  labs(title = "Comparison between Conditions",
       x = "Condition",
       y = "Value") +
  theme_minimal()

# Adding significance annotations using geom_signif with manual mapping
p <- p + geom_signif(comparisons = p_values,
                     annotations = paste0("p = ", round(p_values$p.value, 4)),
                     y_position = 0.0005)  # Adjust as necessary

# Display the plot
print(p)
