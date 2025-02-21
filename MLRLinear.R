#William Schmitt
#2/3/2025
#machine learning practice

library(caret)
library(ggplot2)


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


modellinear <- lm(quality ~ ., data = train_data)
summary(modellinear)



# Get predictions from the linear model
predictions <- predict(modellinear, test_data)

# Create a dataframe with actual and predicted values
plot_data <- data.frame(Actual = test_data$quality, Predicted = predictions)

ggplot(plot_data, aes(x = Actual, y = Predicted)) +
  geom_point(color = 'skyblue') +
  geom_smooth(method = "lm", color = 'red', se = FALSE) +
  ggtitle('Actual vs Predicted Values') +
  xlab('Actual Quality') +
  ylab('Predicted Quality') +
  theme_minimal()

summary(modellinear)


###################################################################

# Calculate MAE and RMSE
mae_lm_pca <- mean(abs(predictions - test_data$quality))
rmse_lm_pca <- sqrt(mean((predictions - test_data$quality)^2))

print(paste("Linear Regression with PCA - Mean Absolute Error (MAE):", mae_lm_pca))
print(paste("Linear Regression with PCA - Root Mean Square Error (RMSE):", rmse_lm_pca))



###################################################################