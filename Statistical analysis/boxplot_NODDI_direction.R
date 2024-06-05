
# Load necessary libraries
library(tidyr)
library(ggplot2)
library(ggsignif)  # For adding significance annotations

# Convert the data to a long format
data_long <- pivot_longer(data, cols = c(X32,X64,X100,X128), names_to = "condition", values_to = "value")

# Remove "X" prefix from condition names
data_long$condition <- gsub("X", "", data_long$condition)

condition_order <- c("32","64","100", "128")

data_long$condition <- factor(data_long$condition, levels = condition_order)

# Create the boxplot
p <- ggplot(data_long, aes(x = condition, y = value, fill = condition)) +
  geom_boxplot(outlier.shape = NA) +  # Removes the outliers from the plot
  labs(x = "Number of Directions for the shell bvalue = 5000",
       y = "fintra") +
  ylim(0.6, 0.85) +  # Sets the limits of the y-axis
  theme_minimal()+
  theme(axis.text.x = element_text(size = 12),
        axis.text.y = element_text(size = 12),  
        axis.title.x = element_text(size = 14),
        axis.title.y = element_text(size = 14),
        legend.position = "none") +  # Remove the legend
  guides(fill = FALSE)  # Remove the fill legend

# Add significance annotations
p <- p + geom_signif(comparisons = list(c("32", "64")),
                     annotations = c("***"),
                     y_position = c(0.75),
                     tip_length = 0.02)  # Adjust as necessary

p <- p + geom_signif(comparisons = list(c("32", "100")),
                     annotations = c("***"),
                     y_position = c(0.78),
                     tip_length = 0.02)  # Adjust as necessary


p <- p + geom_signif(comparisons = list(c("32", "128")),
                     annotations = c("***"),
                     y_position = c(0.83),
                     tip_length = 0.02)  # Adjust as necessary

p <- p + geom_signif(comparisons = list(c("64", "100")),
                     annotations = c("***"),
                     y_position = c(0.76),
                     tip_length = 0.02)  # Adjust as necessary


p <- p + geom_signif(comparisons = list(c("64", "128")),
                     annotations = c("***"),
                     y_position = c(0.81),
                     tip_length = 0.02)  # Adjust as necessary


p <- p + geom_signif(comparisons = list(c("100", "128")),
                     annotations = c("***"),
                     y_position = c(0.77),
                     tip_length = 0.02)  # Adjust as necessary



print(p)

#############################################

# Load necessary libraries
library(tidyr)
library(ggplot2)
library(ggsignif)  # For adding significance annotations

# Convert the data to a long format
data_long <- pivot_longer(data, cols = c(X128,X200,X256), names_to = "condition", values_to = "value")

# Remove "X" prefix from condition names
data_long$condition <- gsub("X", "", data_long$condition)

condition_order <- c("128","200","256")

data_long$condition <- factor(data_long$condition, levels = condition_order)

# Create the boxplot
p <- ggplot(data_long, aes(x = condition, y = value, fill = condition)) +
  geom_boxplot(outlier.shape = NA) +  # Removes the outliers from the plot
  labs(x = "Number of Directions for the shell bvalue = 10000",
       y = "fintra") +
  ylim(0.25, 0.8) +  # Sets the limits of the y-axis
  theme_minimal()+
  theme(axis.text.x = element_text(size = 12),
        axis.text.y = element_text(size = 12),  
        axis.title.x = element_text(size = 14),
        axis.title.y = element_text(size = 14))

# Add significance annotations
p <- p + geom_signif(comparisons = list(c("128", "200")),
                     annotations = c("ns"),
                     y_position = c(0.6),
                     tip_length = 0.02)  # Adjust as necessary

# Add significance annotations
p <- p + geom_signif(comparisons = list(c("128", "256")),
                     annotations = c("**"),
                     y_position = c(0.65),
                     tip_length = 0.02)  # Adjust as necessary

# Add significance annotations
p <- p + geom_signif(comparisons = list(c("200","256")),
                     annotations = c("***"),
                     y_position = c(0.62),
                     tip_length = 0.02)  # Adjust as necessary


print(p)

##############################################################################


# Load necessary libraries
library(tidyr)
library(ggplot2)
library(ggsignif)  # For adding significance annotations

# Convert the data to a long format
data_long <- pivot_longer(data, cols = c(X16,X32,X40,X48,X64), names_to = "condition", values_to = "value")

# Remove "X" prefix from condition names
data_long$condition <- gsub("X", "", data_long$condition)

condition_order <- c("16","32","40","48","64")

data_long$condition <- factor(data_long$condition, levels = condition_order)

# Create the boxplot
p <- ggplot(data_long, aes(x = condition, y = value, fill = condition)) +
  geom_boxplot(outlier.shape = NA) +  # Removes the outliers from the plot
  labs(x = "Number of Directions for the shell bvalue = 1000",
       y = "fiso") +
  ylim(0.00, 0.52) +  # Sets the limits of the y-axis
  theme_minimal()+
  theme(axis.text.x = element_text(size = 12),
        axis.text.y = element_text(size = 12),  
        axis.title.x = element_text(size = 14),
        axis.title.y = element_text(size = 14),
        legend.position = "none") +  # Remove the legend
  guides(fill = FALSE)  # Remove the fill legend

# Add significance annotations
p <- p + geom_signif(comparisons = list(c("16", "32")),
                     annotations = c("*"),
                     y_position = c(0.30),
                     tip_length = 0.02)  # Adjust as necessary

p <- p + geom_signif(comparisons = list(c("16", "40")),
                     annotations = c("*"),
                     y_position = c(0.34),
                     tip_length = 0.02)  # Adjust as necessary


p <- p + geom_signif(comparisons = list(c("16","48")),
                     annotations = c("***"),
                     y_position = c(0.425),
                     tip_length = 0.02)  # Adjust as necessary

p <- p + geom_signif(comparisons = list(c("16","64")),
                     annotations = c("***"),
                     y_position = c(0.48),
                     tip_length = 0.02)  # Adjust as necessary



p <- p + geom_signif(comparisons = list(c("32","40")),
                     annotations = c("ns"),
                     y_position = c(0.31),
                     tip_length = 0.02)  # Adjust as necessary

p <- p + geom_signif(comparisons = list(c("32","48")),
                     annotations = c("ns"),
                     y_position = c(0.37),
                     tip_length = 0.02)  # Adjust as necessary


p <- p + geom_signif(comparisons = list(c("32","64")),
                     annotations = c("ns"),
                     y_position = c(0.45),
                     tip_length = 0.02)  # Adjust as necessary

p <- p + geom_signif(comparisons = list(c("40", "48")),
                     annotations = c("ns"),
                     y_position = c(0.32),
                     tip_length = 0.02)  # Adjust as necessary

p <- p + geom_signif(comparisons = list(c("40", "64")),
                     annotations = c("ns"),
                     y_position = c(0.398),
                     tip_length = 0.02)  # Adjust as necessary


p <- p + geom_signif(comparisons = list(c("48", "64")),
                     annotations = c("ns"),
                     y_position = c(0.33),
                     tip_length = 0.02)  # Adjust as necessary
print(p)
