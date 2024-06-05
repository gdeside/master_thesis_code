
################################################################################################################################################

# Load necessary libraries
library(tidyr)
library(ggplot2)
library(ggsignif)  # For adding significance annotations

# Convert the data to a long format
data_long <- pivot_longer(data, cols = c(X16, X32, X40, X48, X64), names_to = "condition", values_to = "value")

# Remove "X" prefix from condition names
data_long$condition <- gsub("X", "", data_long$condition)



# Create the boxplot
p <- ggplot(data_long, aes(x = condition, y = value, fill = condition)) +
  geom_boxplot(outlier.shape = NA) +  # Removes the outliers from the plot
  labs(x = "Number of Directions for the shell bvalue = 3000",
       y = "RD") +
  ylim(3e-4,0.95e-3) +  # Sets the limits of the y-axis
  theme_minimal()+
  theme(axis.text.x = element_text(size = 12),
        axis.text.y = element_text(size = 12),  
        axis.title.x = element_text(size = 14),
        axis.title.y = element_text(size = 14),
        legend.position = "none") +  # Remove the legend
  guides(fill = FALSE) 

# Add significance annotations
p <- p + geom_signif(comparisons = list(c("16", "32")),
                     annotations = c("***"),
                     y_position = c(0.63e-3))  # Adjust as necessary


p <- p + geom_signif(comparisons = list(c("16", "40")),
                     annotations = c("***"),
                     y_position = c(0.69e-3))  # Adjust as necessary

p <- p + geom_signif(comparisons = list(c("16", "48")),
                     annotations = c("***"),
                     y_position = c(0.81e-3))  # Adjust as necessary


p <- p + geom_signif(comparisons = list(c("16", "64")),
                     annotations = c("***"),
                     y_position = c(0.88e-3))  # Adjust as necessary

p <- p + geom_signif(comparisons = list(c("32", "40")),
                     annotations = c("***"),
                     y_position = c(0.65e-3))  # Adjust as necessary

p <- p + geom_signif(comparisons = list(c("32", "48")),
                     annotations = c("***"),
                     y_position = c(0.73e-3))  # Adjust as necessary

p <- p + geom_signif(comparisons = list(c("32", "64")),
                     annotations = c("***"),
                     y_position = c(0.84e-3))  # Adjust as necessary


p <- p + geom_signif(comparisons = list(c("40", "48")),
                     annotations = c("***"),
                     y_position = c(0.67e-3))  # Adjust as necessary

p <- p + geom_signif(comparisons = list(c("40", "64")),
                     annotations = c("***"),
                     y_position = c(0.77e-3))  # Adjust as necessary

p <- p + geom_signif(comparisons = list(c("48", "64")),
                     annotations = c("***"),
                     y_position = c(0.69e-3))  # Adjust as necessary

print(p)