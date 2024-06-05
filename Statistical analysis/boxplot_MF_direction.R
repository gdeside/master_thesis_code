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
  labs(title = "Comparison between Conditions",
       x = "Number of Directions for the shell bvalue = 10000",
       y = "fvf_tot") +
  ylim(0.3, 0.5) +  # Sets the limits of the y-axis
  theme_minimal()+
  theme(axis.text.x = element_text( size = 12,),
        axis.text.y = element_text(size = 12),  
        axis.title.x = element_text(size = 14),
        axis.title.y = element_text(size = 14))

# Add significance annotations
p <- p + geom_signif(comparisons = list(c("128", "200")),
                     annotations = c("***"),
                     y_position = c(0.34))  # Adjust as necessary

# Add significance annotations
p <- p + geom_signif(comparisons = list(c("128", "256")),
                     annotations = c("***"),
                     y_position = c(0.365))  # Adjust as necessary

# Add significance annotations
p <- p + geom_signif(comparisons = list(c("200","256")),
                     annotations = c("*"),
                     y_position = c(0.35))  # Adjust as necessary


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
  labs(x = "Number of Directions for the shell bvalue = 3000",
       y = "frac_f0") +
  ylim(0.55, 0.97) +  # Sets the limits of the y-axis
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
                     y_position = c(0.785))  # Adjust as necessary


p <- p + geom_signif(comparisons = list(c("16", "40")),
                     annotations = c("***"),
                     y_position = c(0.82))  # Adjust as necessary


p <- p + geom_signif(comparisons = list(c("16","48")),
                     annotations = c("***"),
                     y_position = c(0.89))  # Adjust as necessary

p <- p + geom_signif(comparisons = list(c("16","64")),
                     annotations = c("***"),
                     y_position = c(0.95))  # Adjust as necessary


p <- p + geom_signif(comparisons = list(c("32","40")),
                     annotations = c("ns"),
                     y_position = c(0.795))  # Adjust as necessary

p <- p + geom_signif(comparisons = list(c("32","48")),
                     annotations = c("ns"),
                     y_position = c(0.845))  # Adjust as necessary

p <- p + geom_signif(comparisons = list(c("32","64")),
                     annotations = c("***"),
                     y_position = c(0.92))  # Adjust as necessary

p <- p + geom_signif(comparisons = list(c("40", "48")),
                     annotations = c("ns"),
                     y_position = c(0.805))  # Adjust as necessary

p <- p + geom_signif(comparisons = list(c("40", "64")),
                     annotations = c("***"),
                     y_position = c(0.865))  # Adjust as necessary

p <- p + geom_signif(comparisons = list(c("48", "64")),
                     annotations = c("***"),
                     y_position = c(0.82))  # Adjust as necessary



print(p)


#############################################

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
       y = "fvf_tot") +
  ylim(0.35, 0.6) +  # Sets the limits of the y-axis
  theme_minimal()+
  theme(axis.text.x = element_text(size = 12),
        axis.text.y = element_text(size = 12),  
        axis.title.x = element_text(size = 14),
        axis.title.y = element_text(size = 14),
        legend.position = "none") +  # Remove the legend
  guides(fill = FALSE)  # Remove the fill legend

# Add significance annotations
p <- p + geom_signif(comparisons = list(c("32", "64")),
                     annotations = c("*"),
                     y_position = c(0.45))  # Adjust as necessary


p <- p + geom_signif(comparisons = list(c("32", "100")),
                     annotations = c("***"),
                     y_position = c(0.48))  # Adjust as necessary


p <- p + geom_signif(comparisons = list(c("32", "128")),
                     annotations = c("***"),
                     y_position = c(0.52))  # Adjust as necessary

p <- p + geom_signif(comparisons = list(c("64", "100")),
                     annotations = c("*"),
                     y_position = c(0.46))  # Adjust as necessary


p <- p + geom_signif(comparisons = list(c("64", "128")),
                     annotations = c("***"),
                     y_position = c(0.50))  # Adjust as necessary


p <- p + geom_signif(comparisons = list(c("100", "128")),
                     annotations = c("*"),
                     y_position = c(0.47))  # Adjust as necessary

print(p)

