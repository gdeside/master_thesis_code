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
  labs(x = "Shells used",
       y = "fiso") +
  ylim(0.0, 0.8) +  # Sets the limits of the y-axis
  theme_minimal()+
  theme(axis.text.x = element_text(angle = 90, vjust = 0.5))+
  theme(axis.text.x = element_text(size = 12),
        axis.text.y = element_text(size = 12),  
        axis.title.x = element_text(size = 14),
        axis.title.y = element_text(size = 14),
        legend.position = "none") +  # Remove the legend
  guides(fill = FALSE)  # Remove the fill legend


# Add significance annotations
p <- p + geom_signif(comparisons = list(c("b1000+b3000", "b1000+b5000")),
                     annotations = c("ns"),
                     y_position = 0.3,
                     size = 0.5,
                     textsize = 3.5,
                     tip_length = 0.02,
                     vjust=0.2)  # Adjust as necessary


p <- p + geom_signif(comparisons = list(c("b1000+b3000", "b1000+b10000")),
                     annotations = c("***"),
                     y_position = 0.4,
                     size = 0.5,
                     textsize = 3.5,
                     tip_length = 0.02,
                     vjust=0.7)  # Adjust as necessary


p <- p + geom_signif(comparisons = list(c("b1000+b3000", "b3000+b5000")),
                     annotations = c("***"),
                     y_position = 0.52,
                     size = 0.5,
                     textsize = 3.5,
                     tip_length = 0.02,
                     vjust=0.7)  # Adjust as necessary


p <- p + geom_signif(comparisons = list(c("b1000+b3000", "b3000+b10000")),
                     annotations = c("***"),
                     y_position = 0.62,
                     size = 0.5,
                     textsize = 3.5,
                     tip_length = 0.02,
                     vjust=0.7)  # Adjust as necessary

p <- p + geom_signif(comparisons = list(c("b1000+b3000", "b5000+b10000")),
                     annotations = c("***"),
                     y_position = 0.7,
                     size = 0.5,
                     textsize = 3.5,
                     tip_length = 0.02,
                     vjust=0.7)  # Adjust as necessary

p <- p + geom_signif(comparisons = list(c("b1000+b3000", "all shells")),
                     annotations = c("**"),
                     y_position = 0.76,
                     size = 0.5,
                     textsize = 3.5,
                     tip_length = 0.02,
                     vjust=0.7)  # Adjust as necessary

##
p <- p + geom_signif(comparisons = list(c("b1000+b5000", "b1000+b10000")),
                     annotations = c("*"),
                     y_position = 0.32,
                     size = 0.5,
                     textsize = 3.5,
                     tip_length = 0.02,
                     vjust=0.7)  # Adjust as necessary


p <- p + geom_signif(comparisons = list(c("b1000+b5000", "b3000+b5000")),
                     annotations = c("***"),
                     y_position = 0.43,
                     size = 0.5,
                     textsize = 3.5,
                     tip_length = 0.02,
                     vjust=0.7)  # Adjust as necessary

p <- p + geom_signif(comparisons = list(c("b1000+b5000", "b3000+b10000")),
                     annotations = c("***"),
                     y_position = 0.54,
                     size = 0.5,
                     textsize = 3.5,
                     tip_length = 0.02,
                     vjust=0.7)  # Adjust as necessary


p <- p + geom_signif(comparisons = list(c("b1000+b5000", "b5000+b10000")),
                     annotations = c("***"),
                     y_position = 0.645,
                     size = 0.5,
                     textsize = 3.5,
                     tip_length = 0.02,
                     vjust=0.7)  # Adjust as necessary

p <- p + geom_signif(comparisons = list(c("b1000+b5000", "all shells")),
                     annotations = c("ns"),
                     y_position = 0.73,
                     size = 0.5,
                     textsize = 3.5,
                     tip_length = 0.02,
                     vjust=0.2)  # Adjust as necessary
##

p <- p + geom_signif(comparisons = list(c("b1000+b10000", "b3000+b5000")),
                     annotations = c("***"),
                     y_position = 0.35,
                     size = 0.5,
                     textsize = 3.5,
                     tip_length = 0.02,
                     vjust=0.7)  # Adjust as necessary


p <- p + geom_signif(comparisons = list(c("b1000+b10000", "b3000+b10000")),
                     annotations = c("***"),
                     y_position = 0.46,
                     size = 0.5,
                     textsize = 3.5,
                     tip_length = 0.02,
                     vjust=0.7)  # Adjust as necessary

p <- p + geom_signif(comparisons = list(c("b1000+b10000", "b5000+b10000")),
                     annotations = c("ns"),
                     y_position = 0.57,
                     size = 0.5,
                     textsize = 3.5,
                     tip_length = 0.02,
                     vjust=0.2)  # Adjust as necessary



p <- p + geom_signif(comparisons = list(c("b1000+b10000", "all shells")),
                     annotations = c("ns"),
                     y_position = 0.67,
                     size = 0.5,
                     textsize = 3.5,
                     tip_length = 0.02,
                     vjust=0.2)  # Adjust as necessary
##


p <- p + geom_signif(comparisons = list(c("b3000+b5000", "b3000+b10000")),
                     annotations = c("ns"),
                     y_position = 0.37,
                     size = 0.5,
                     textsize = 3.5,
                     tip_length = 0.02,
                     vjust=0.2)  # Adjust as necessary

p <- p + geom_signif(comparisons = list(c("b3000+b5000", "b5000+b10000")),
                     annotations = c("*"),
                     y_position = 0.49,
                     size = 0.5,
                     textsize = 3.5,
                     tip_length = 0.02,
                     vjust=0.7)  # Adjust as necessary

p <- p + geom_signif(comparisons = list(c("b3000+b5000", "all shells")),
                     annotations = c("***"),
                     y_position = 0.6,
                     size = 0.5,
                     textsize = 3.5,
                     tip_length = 0.02,
                     vjust=0.7)  # Adjust as necessary


##

p <- p + geom_signif(comparisons = list(c("b3000+b10000", "b5000+b10000")),
                     annotations = c("***"),
                     y_position = 0.39,
                     size = 0.5,
                     textsize = 3.5,
                     tip_length = 0.02,
                     vjust=0.7)  # Adjust as necessary


p <- p + geom_signif(comparisons = list(c("b3000+b10000", "all shells")),
                     annotations = c("***"),
                     y_position = 0.52,
                     size = 0.5,
                     textsize = 3.5,
                     tip_length = 0.02,
                     vjust=0.7)  # Adjust as necessary
##
p <- p + geom_signif(comparisons = list(c("b5000+b10000", "all shells")),
                     annotations = c("*"),
                     y_position = 0.41,
                     size = 0.5,
                     textsize = 3.5,
                     tip_length = 0.02,
                     vjust=0.7)  # Adjust as necessary


print(p)