#William Schmitt
#2/18/2025
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


# Standardize the data (excluding the 'quality' column)
df_standardized <- as.data.frame(scale(df[, -ncol(df)]))
# Perform PCA
pca_result <- prcomp(df_standardized, center = TRUE, scale. = TRUE)
# Print summary of PCA results
summary(pca_result)
# Scree Plot to visualize explained variance
explained_variance <- pca_result$sdev^2 / sum(pca_result$sdev^2) * 100
plot(explained_variance, type = 'b', main = 'Scree Plot', xlab = 'Principal Component', ylab = 'Explained Variance (%)')
# Biplot
biplot(pca_result, scale = 0)

# Create a data frame with the principal components
pca_data <- data.frame(pca_result$x)
# Add the 'quality' column back to the PCA data frame for analysis
pca_data$quality <- df$quality
# Visualize the first two principal components
ggplot(pca_data, aes(x = PC1, y = PC2, color = as.factor(quality))) +
  geom_point() +
  labs(title = 'PCA of Wine Quality Data', x = 'Principal Component 1', y = 'Principal Component 2', color = 'Quality') +
  theme_minimal()

# Extract the top principal components (e.g., first 5 components)
num_components <- 10
principal_components <- pca_result$x[, 1:num_components]
print(pca_result$x[, 1:num_components])
# Combine the principal components with the 'quality' column for modeling
pca_data <- data.frame(principal_components, quality = df$quality)
print(pca_data)
print(pca_result)
print(pca_result$x)
print(explained_variance)
set.seed(123)

#####################

#Removing insugnificant componets of PCA
pca_data <- pca_data[, -c(6, 7)]

#####################



# Create an 80-20 train-test split
train_index <- createDataPartition(pca_data$quality, p = 0.8, list = FALSE)
train_data <- pca_data[train_index, ]
test_data <- pca_data[-train_index, ]

# Print the dimensions of the train and test sets
print(dim(train_data))
print(dim(test_data))



# Fit logistic regression model
model_lm_pca <- lm(quality ~ ., data = train_data)

# Print model summary
summary(model_lm_pca)

# Predict on test data
predictions_lm_pca <- predict(model_lm_pca, test_data)
print(predictions_lm_pca)

# Calculate MAE and RMSE
mae_lm_pca <- mean(abs(predictions_lm_pca - test_data$quality))
rmse_lm_pca <- sqrt(mean((predictions_lm_pca - test_data$quality)^2))

print(paste("Linear Regression with PCA - Mean Absolute Error (MAE):", mae_lm_pca))
print(paste("Linear Regression with PCA - Root Mean Square Error (RMSE):", rmse_lm_pca))

############################################################################################

# Assuming you have the logistic regression predictions with PCA
predictions_lm_pca <- predict(model_lm_pca, test_data)
plot_data_pca <- data.frame(Actual = test_data$quality, Predicted = predictions_lm_pca)


summary(model_lm_pca)$coefficients

p1 <- ggplot(plot_data_pca, aes(x = test_data$quality, y = predictions_lm_pca)) +
  geom_point(color = 'skyblue') +
  geom_smooth(method = "lm", color = 'red', se = FALSE) +
  ggtitle('PCA Data: Actual vs Predicted Values') +
  xlab('Actual Quality') +
  ylab('Predicted Quality') +
  theme_minimal()

print(p1)


