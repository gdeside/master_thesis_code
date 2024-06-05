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
  labs(title = "Comparison between Conditions",
       x = "Number of Directions for the shell bvalue = 5000",
       y = "Intra-cellular volume fraction (fintra)") +
  ylim(0.3, 0.75) +  # Sets the limits of the y-axis
  theme_minimal()

# Add significance annotations
p <- p + geom_signif(comparisons = list(c("32", "64")),
                     annotations = c("***"),
                     y_position = c(0.6))  # Adjust as necessary

p <- p + geom_signif(comparisons = list(c("32", "100")),
                     annotations = c("***"),
                     y_position = c(0.63))  # Adjust as necessary

p <- p + geom_signif(comparisons = list(c("32", "128")),
                     annotations = c("***"),
                     y_position = c(0.68))  # Adjust as necessary

p <- p + geom_signif(comparisons = list(c("64", "100")),
                     annotations = c("***"),
                     y_position = c(0.61))  # Adjust as necessary

p <- p + geom_signif(comparisons = list(c("64", "128")),
                     annotations = c("***"),
                     y_position = c(0.655))  # Adjust as necessary

p <- p + geom_signif(comparisons = list(c("100", "128")),
                     annotations = c("***"),
                     y_position = c(0.62))  # Adjust as necessary


print(p)

######################################################

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
  labs(title = "Comparison between Conditions",
       x = "Number of Directions for the shell bvalue = 3000",
       y = "Fiber bundles volume fraction (fbundle)") +
  ylim(0.8, 1.1) +  # Sets the limits of the y-axis
  theme_minimal()  +
  theme(axis.text.x = element_text(size = 12),  # Adjust x-axis text size
        axis.text.y = element_text(size = 12))  # Adjust y-axis text size

# Add significance annotations
p <- p + geom_signif(comparisons = list(c("16", "32")),
                     annotations = c("ns"),
                     y_position = c(0.93))  # Adjust as necessary

p <- p + geom_signif(comparisons = list(c("16", "40")),
                     annotations = c("ns"),
                     y_position = c(0.948))  # Adjust as necessary

p <- p + geom_signif(comparisons = list(c("16", "48")),
                     annotations = c("ns"),
                     y_position = c(1))  # Adjust as necessary


p <- p + geom_signif(comparisons = list(c("16", "64")),
                     annotations = c("***"),
                     y_position = c(1.04))  # Adjust as necessary


p <- p + geom_signif(comparisons = list(c("32", "40")),
                     annotations = c("ns"),
                     y_position = c(0.935))  # Adjust as necessary

p <- p + geom_signif(comparisons = list(c("32", "48")),
                     annotations = c("ns"),
                     y_position = c(0.963))  # Adjust as necessary

p <- p + geom_signif(comparisons = list(c("32", "64")),
                     annotations = c("***"),
                     y_position = c(1.015))  # Adjust as necessary


p <- p + geom_signif(comparisons = list(c("40", "48")),
                     annotations = c("ns"),
                     y_position = c(0.94))  # Adjust as necessary

p <- p + geom_signif(comparisons = list(c("40", "64")),
                     annotations = c("***"),
                     y_position = c(0.978))  # Adjust as necessary


p <- p + geom_signif(comparisons = list(c("48", "64")),
                     annotations = c("***"),
                     y_position = c(0.945))  # Adjust as necessary

print(p)

################################################################

# Load necessary libraries
library(tidyr)
library(ggplot2)
library(ggsignif)  # For adding significance annotations

# Convert the data to a long format
data_long <- pivot_longer(data, cols = c(b1000, b3000, b5000, b10000, all_bvals), names_to = "condition", values_to = "value")

# Remove "X" prefix from condition names
data_long$condition <- gsub("X", "", data_long$condition)

data_long$condition[data_long$condition == "all_bvals"] <- "all shells"

# Define the order of conditions
condition_order <- c("b1000", "b3000", "b5000", "b10000", "all shells")

# Convert condition to factor with defined order
data_long$condition <- factor(data_long$condition, levels = condition_order)

z = 0.87
# Create the boxplot
p <- ggplot(data_long, aes(x = condition, y = value, fill = condition)) +
  geom_boxplot(outlier.shape = NA) +  # Removes the outliers from the plot
  labs(title = "Comparison between Conditions",
       x = "Shell used",
       y = "Extra-cellular volume fraction (fextra)") +
  ylim(0.0, z) +  # Sets the limits of the y-axis
  theme_minimal()

# Add significance annotations
p <- p + geom_signif(comparisons = list(c("b1000", "b3000")),
                     annotations = c("***"),
                     y_position = c(0.5))  # Adjust as necessary

p <- p + geom_signif(comparisons = list(c("b1000", "b5000")),
                     annotations = c("***"),
                     y_position = c(0.555))  # Adjust as necessary

p <- p + geom_signif(comparisons = list(c("b1000", "b10000")),
                     annotations = c("***"),
                     y_position = c(0.655))  # Adjust as necessary



p <- p + geom_signif(comparisons = list(c("b1000", "all shells")),
                     annotations = c("***"),
                     y_position = c(0.725))  # Adjust as necessary




p <- p + geom_signif(comparisons = list(c("b3000", "b5000")),
                     annotations = c("***"),
                     y_position = c(0.515))  # Adjust as necessary

p <- p + geom_signif(comparisons = list(c("b3000", "b10000")),
                     annotations = c("ns"),
                     y_position = c(0.59))  # Adjust as necessary


p <- p + geom_signif(comparisons = list(c("b3000", "all shells")),
                     annotations = c("***"),
                     y_position = c(0.69))  # Adjust as necessary


p <- p + geom_signif(comparisons = list(c("b5000", "b10000")),
                     annotations = c("***"),
                     y_position = c(0.525))  # Adjust as necessary


p <- p + geom_signif(comparisons = list(c("b5000", "all shells")),
                     annotations = c("***"),
                     y_position = c(0.62))  # Adjust as necessary

p <- p + geom_signif(comparisons = list(c("b10000", "all shells")),
                     annotations = c("***"),
                     y_position = c(0.535))  # Adjust as necessary

print(p)


########################################################################
# Load necessary libraries
library(tidyr)
library(ggplot2)
library(ggsignif)  # For adding significance annotations

# Convert the data to a long format
data_long <- pivot_longer(data, cols = c("X.1000..3000.","X.1000..5000.","X.1000..10000.","X.3000..5000.","X.3000..10000.","X.5000..10000.", "all_bvals"), names_to = "condition", values_to = "value")

data_long$condition <- ifelse(data_long$condition == "X.1000..3000.", "b1000+b3000",
                              ifelse(data_long$condition == "X.1000..5000.", "b1000+b5000",
                                     ifelse(data_long$condition == "X.1000..10000.", "b1000+b10000",
                                            ifelse(data_long$condition == "X.3000..5000.", "b3000+b5000",
                                                   ifelse(data_long$condition == "X.3000..10000.", "b3000+b10000",
                                                          ifelse(data_long$condition == "X.5000..10000.", "b5000+b10000",
                                                                 "all shells"))))))

condition_order <- c("b1000+b3000","b1000+b5000","b1000+b10000","b3000+b5000","b3000+b10000","b5000+b10000","all shells")

data_long$condition <- factor(data_long$condition, levels = condition_order)
# Create the boxplot
p <- ggplot(data_long, aes(x = condition, y = value, fill = condition)) +
  geom_boxplot(outlier.shape = NA) +  # Removes the outliers from the plot
  labs(title = "Comparison between Conditions",
       x = "Shells used",
       y = "Axial diffusivity (AD)") +
  ylim(0.65, 1.1) +  # Sets the limits of the y-axis
  theme_minimal()+
  theme(axis.text.x = element_text(angle = 90, vjust = 0.5))

# Add significance annotations
p <- p + geom_signif(comparisons = list(c("b1000+b3000", "b1000+b5000")),
                     annotations = c("***"),
                     y_position = 0.92)  # Adjust as necessary


p <- p + geom_signif(comparisons = list(c("b1000+b3000", "b1000+b10000")),
                     annotations = c("***"),
                     y_position = 0.959)  # Adjust as necessary

p <- p + geom_signif(comparisons = list(c("b1000+b5000", "b1000+b10000")),
                     annotations = c("***"),
                     y_position = 0.935)  # Adjust as necessary

p <- p + geom_signif(comparisons = list(c("b1000+b5000", "b3000+b5000")),
                     annotations = c("***"),
                     y_position = 0.987)  # Adjust as necessary


p <- p + geom_signif(comparisons = list(c("b1000+b10000", "b3000+b5000")),
                     annotations = c("***"),
                     y_position = 0.947)  # Adjust as necessary

p <- p + geom_signif(comparisons = list(c("b1000+b10000", "b3000+b10000")),
                     annotations = c("***"),
                     y_position = 1.015)  # Adjust as necessary

p <- p + geom_signif(comparisons = list(c("b3000+b5000", "b3000+b10000")),
                     annotations = c("***"),
                     y_position = 0.959)  # Adjust as necessary

p <- p + geom_signif(comparisons = list(c("b3000+b5000", "b5000+b10000")),
                     annotations = c("***"),
                     y_position = 1.048)  # Adjust as necessary


p <- p + geom_signif(comparisons = list(c("b3000+b10000", "b5000+b10000")),
                     annotations = c("***"),
                     y_position = 0.97)  # Adjust as necessary

p <- p + geom_signif(comparisons = list(c("b3000+b10000", "all shells")),
                     annotations = c("***"),
                     y_position = 1.075)  # Adjust as necessary


p <- p + geom_signif(comparisons = list(c("b5000+b10000", "all shells")),
                     annotations = c("***"),
                     y_position = 0.982)  # Adjust as necessary

print(p)


