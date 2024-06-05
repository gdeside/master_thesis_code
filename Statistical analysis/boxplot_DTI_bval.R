##DTI bvals

# Load necessary libraries
library(tidyr)
library(ggplot2)
library(ggsignif)  # For adding significance annotations

# Convert the data to a long format
data_long <- pivot_longer(data, cols = c(b1000, b3000, all_bvals), names_to = "condition", values_to = "value")

# Remove "X" prefix from condition names
data_long$condition <- gsub("X", "", data_long$condition)

data_long$condition[data_long$condition == "all_bvals"] <- "all shells"

# Define the order of conditions
condition_order <- c("b1000", "b3000", "all shells")

# Convert condition to factor with defined order
data_long$condition <- factor(data_long$condition, levels = condition_order)


# Create the boxplot
p <- ggplot(data_long, aes(x = condition, y = value, fill = condition)) +
  geom_boxplot(outlier.shape = NA) +  # Removes the outliers from the plot
  labs(x = "Shell used",
       y = "FA") +
  ylim(0.4, 0.8) +  # Sets the limits of the y-axis
  theme_minimal()+
  theme(axis.text.x = element_text(size = 12),
        axis.text.y = element_text(size = 12),  
        axis.title.x = element_text(size = 14),
        axis.title.y = element_text(size = 14),
        legend.position = "none") +  # Remove the legend
  guides(fill = FALSE)  # Remove the fill legend


# Add significance annotations
p <- p + geom_signif(comparisons = list(c("b1000", "b3000")),
                     annotations = c("***"),
                     y_position = c(0.65))  # Adjust as necessary

p <- p + geom_signif(comparisons = list(c("b1000", "all shells")),
                     annotations = c("ns"),
                     y_position = c(0.75))  # Adjust as necessary

p <- p + geom_signif(comparisons = list(c("b3000", "all shells")),
                     annotations = c("***"),
                     y_position = c(0.7))  # Adjust as necessary

print(p)