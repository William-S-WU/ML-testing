#William Schmitt
#2/3/2025
#machine learning practice

library(ggplot2)

getwd()
print(getwd())
current_directory <- getwd()
df <- read.csv("wine-quality.csv")
column_names <- colnames(df)
print(column_names)
anova <- aov(quality ~ ., data = df)
summary(anova)

modellinear <- lm(quality ~ ., data = df)
summary(modellinear)



# Get predictions from the linear model
predictions <- predict(modellinear, df)

# Create a dataframe with actual and predicted values
plot_data <- data.frame(Actual = df$quality, Predicted = predictions)

ggplot(plot_data, aes(x = Actual, y = Predicted)) +
  geom_point(color = 'skyblue') +
  geom_smooth(method = "lm", color = 'red', se = FALSE) +
  ggtitle('Actual vs Predicted Values') +
  xlab('Actual Quality') +
  ylab('Predicted Quality') +
  theme_minimal()

summary(modellinear)

modellog <- glm(quality ~ ., data = df)

summary(modellog)
predictions2 <- predict(modellog, df)

plot_data2 <- data.frame(Actual = df$quality, Predicted = predictions2)

ggplot(plot_data2, aes(x = Actual, y = Predicted)) +
  geom_point(color = 'skyblue') +
  geom_smooth(method = "lm", color = 'red', se = FALSE) +
  ggtitle('Actual vs Predicted Values') +
  xlab('Actual Quality') +
  ylab('Predicted Quality') +
  theme_minimal()

