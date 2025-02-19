#William Schmitt
#2/3/2025
#machine learning practice

library(caret)
library(ggplot2)
library(nnet)

getwd()
print(getwd())
current_directory <- getwd()
df <- read.csv("wine-quality.csv")
column_names <- colnames(df)
print(column_names)
anova <- aov(quality ~ ., data = df)
summary(anova)

set.seed(123)

# Create an 80-20 train-test split
train_index <- createDataPartition(df$quality, p = 0.8, list = FALSE)
train_data <- df[train_index, ]
test_data <- df[-train_index, ]

# Print the dimensions of the train and test sets
print(dim(train_data))
print(dim(test_data))




modellog <- multinom(quality ~ ., data = train_data)

summary(modellog)
predictions <- predict(modellog, test_data)
print(predictions)

actual <- test_data$quality

predictions_numeric <- as.numeric(as.character(predictions))
actual_numeric <- as.numeric(as.character(test_data$quality))

plot_data2 <- data.frame(Actual = test_data$quality, Predicted = predictions)

ggplot(plot_data2, aes(x = Actual, y = Predicted)) +
  geom_point(color = 'skyblue') +
  geom_smooth(method = "lm", color = 'red', se = FALSE) +
  ggtitle('Actual vs Predicted Values') +
  xlab('Actual Quality') +
  ylab('Predicted Quality') +
  theme_minimal()

###################################################################

# Assuming you have the actual and predicted values in plot_data2


# Calculate MAE
mae <- mean(abs(predictions_numeric - actual_numeric))
print(paste("Mean Absolute Error (MAE):", mae))
# Calculate RMSE
rmse <- sqrt(mean((predictions_numeric - actual_numeric)^2))
print(paste("Root Mean Square Error (RMSE):", rmse))

###################################################################

