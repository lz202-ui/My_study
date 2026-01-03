#############################################
# SS9055B Final Project: GLM Analysis on Bank Marketing Dataset
# Author: Lina Zeng
# Date: 2025-04-06
#############################################

# --------------------------
# 1. Load Required Packages
# --------------------------
install.packages("tidyverse")
install.packages("caret")
install.packages("pROC")
install.packages("MASS")

library(tidyverse)
library(ggplot2)
library(caret)
library(pROC)
library(MASS)

# --------------------------
# 2. Load Data and Initial Exploration
# --------------------------
data <- read.csv("D:/Downloads/bank+marketing/bank/bank-full.csv", sep = ";")
str(data)
summary(data)
anyNA(data)

# --------------------------
# 3. Summary Statistics and Visualization
# --------------------------
target_counts <- table(data$y)
print(target_counts)

ggplot(data, aes(x = y)) +
  geom_bar(fill = "steelblue") +
  labs(title = "Distribution of Subscription Results", x = "Subscription Status (y)", y = "Count") +
  theme_minimal()

ggplot(data, aes(x = age, fill = y)) +
  geom_histogram(bins = 30, position = "stack", color = "black") +
  labs(title = "Age Distribution by Subscription Status", x = "Age", y = "Count") +
  theme_minimal()

ggplot(data, aes(x = job, fill = y)) +
  geom_bar(position = "dodge") +
  labs(title = "Subscription Status by Job", x = "Job", y = "Count") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

ggplot(data, aes(x = marital, fill = y)) +
  geom_bar(position = "dodge") +
  labs(title = "Subscription Status by Marital Status", x = "Marital Status", y = "Count") +
  theme_minimal()

# --------------------------
# 4. Data Preprocessing and GLM Model Building
# --------------------------
data_clean <- data
data_clean$y <- ifelse(data_clean$y == "yes", 1, 0)
categorical_vars <- c("job", "marital", "education", "default", "housing", "loan", "contact", "month", "poutcome")
data_clean[categorical_vars] <- lapply(data_clean[categorical_vars], as.factor)

# --------------------------
# 5. Train-Test Split
# --------------------------
set.seed(123)
train_index <- createDataPartition(data_clean$y, p = 0.8, list = FALSE)
train <- data_clean[train_index, ]
test <- data_clean[-train_index, ]

# --------------------------
# 6. Full Model and Stepwise Selection (AIC)
# --------------------------
full_model <- glm(y ~ ., data = train, family = binomial)
step_model <- stepAIC(full_model, direction = "both", trace = FALSE)
summary(step_model)

# --------------------------
# 7. Prediction and Evaluation (Unweighted)
# --------------------------
prob_pred <- predict(step_model, newdata = test, type = "response")
class_pred <- ifelse(prob_pred > 0.5, 1, 0)
conf_matrix <- confusionMatrix(factor(class_pred), factor(test$y))
print(conf_matrix)

roc_obj <- roc(test$y, prob_pred)
plot(roc_obj, main = "ROC Curve", col = "blue", lwd = 2)
auc_value <- auc(roc_obj)
cat("AUC Value:", auc_value, "\n")

# --------------------------
# 8. Variable Importance Analysis
# --------------------------
coefficients <- coef(step_model)[-1]
odds_ratio <- exp(coefficients)
importance_df <- data.frame(
  Feature = names(coefficients),
  Coefficient = coefficients,
  OddsRatio = odds_ratio
)
importance_df$AbsCoef <- abs(importance_df$Coefficient)
importance_df_sorted <- importance_df[order(-importance_df$AbsCoef), c("Feature", "Coefficient", "OddsRatio")]
print(head(importance_df_sorted, 10))

# --------------------------
# 9. Weighted Logistic Regression
# --------------------------
w <- ifelse(train$y == 1, 1, sum(train$y == 0) / sum(train$y == 1))
weighted_model <- glm(y ~ ., data = train, family = binomial, weights = w)
step_model <- stepAIC(weighted_model, direction = "both", trace = FALSE)
prob_pred <- predict(step_model, newdata = test, type = "response")
class_pred <- ifelse(prob_pred > 0.5, 1, 0)
print(confusionMatrix(factor(class_pred), factor(test$y)))

roc_obj <- roc(test$y, prob_pred)
plot(roc_obj, main = "ROC Curve with Class Weights", col = "darkgreen", lwd = 2)
print(auc(roc_obj))

# --------------------------
# 10. Threshold Comparison Between Weighted and Unweighted Models
# --------------------------
model_plain <- glm(y ~ ., data = train, family = binomial)
step_plain <- stepAIC(model_plain, direction = "both", trace = FALSE)
prob_plain <- predict(step_plain, newdata = test, type = "response")

model_weighted <- glm(y ~ ., data = train, family = binomial, weights = w)
step_weighted <- stepAIC(model_weighted, direction = "both", trace = FALSE)
prob_weighted <- predict(step_weighted, newdata = test, type = "response")

evaluate_thresholds <- function(prob, true_y, label = "Model") {
  thresholds <- seq(0.1, 0.9, by = 0.1)
  results <- data.frame()
  for (th in thresholds) {
    pred <- ifelse(prob > th, 1, 0)
    cm <- confusionMatrix(factor(pred), factor(true_y), positive = "1")
    precision <- cm$byClass["Precision"]
    recall <- cm$byClass["Recall"]
    f1 <- cm$byClass["F1"]
    acc <- cm$overall["Accuracy"]
    results <- rbind(results, data.frame(
      Model = label,
      Threshold = th,
      Accuracy = round(acc, 4),
      Precision = round(precision, 4),
      Recall = round(recall, 4),
      F1 = round(f1, 4)
    ))
  }
  return(results)
}

roc_plain <- roc(test$y, prob_plain)
roc_weighted <- roc(test$y, prob_weighted)
cat("AUC (Unweighted Model):", round(auc(roc_plain), 4), "\n")
cat("AUC (Weighted Model):", round(auc(roc_weighted), 4), "\n")

plot(roc_plain, col = "black", lwd = 2, main = "ROC Curve Comparison")
lines(roc_weighted, col = "blue", lwd = 2)
legend("bottomright", legend = c("Unweighted Model", "Weighted Model"), col = c("black", "blue"), lwd = 2)

eval_plain <- evaluate_thresholds(prob_plain, test$y, "Unweighted Model")
eval_weighted <- evaluate_thresholds(prob_weighted, test$y, "Weighted Model")
eval_all <- rbind(eval_plain, eval_weighted)

print("==== Performance Comparison: Weighted vs Unweighted Models Across Thresholds ====")
print(eval_all)
