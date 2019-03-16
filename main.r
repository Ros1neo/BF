# This dataset is concerning Black Friday sales
# We want to run K-means clustering in order to get the customer profile of the most spending population on Black Friday. K-means clustering on multiple dimensions
# We would like to identify what product might be bought together, which one are the best selling products. Dummy variables on product category and K Means clustering on the products to identify combos to make discounts
# We would like to know which are the clusters that are not spending a lot, but that would potentially spend more, if we change a marketing parameter, ex bundle selling
# We would like to predict the spent amount based on the customer profile, and find the most accurative features predicting this. Regression
# Multiclass logistic regression to identify what kind of bundle could be sold to a specific customer profile

# %% Imports
library(randomForest)
library(readr)
library(dplyr)

data <- read.csv("BlackFriday.csv")
print(head(data))

set.seed(1234)

# %% K means clustering
kmeans(data$Purchase, 5, 10)
train_split <- 0.85
# data <- data[sample(1:nrow(full),nrow(full),replace = F),] %>% select(-PassengerId, Name, Ticket)

#R’s Random Forest algorithm has a few restrictions that we did not have with our decision trees.
#The big one has been the elephant in the room until now, we have to clean up the missing values in our dataset.
#rpart has a great advantage in that it can use surrogate variables when it encounters an NA value.

#Let's use an Decision Tree to fill them !
#Agefit <- rpart( ~ Pclass + Sex + SibSp + Parch + Fare + Embarked,
#                data=full[!is.na(full$Age),],
#                method="anova")

#full$Age[is.na(full$Age)] <- predict(Agefit, full[is.na(full$Age),])

#train <- full[1:floor(train_split*nrow(full)),]
#test <- full[floor(train_split*nrow(full)):nrow(full),]

#as for Embarqued and Cabin let's just forget about them

#train <- train %>% filter(!is.na(Embarked)) %>% select(-Cabin)
#train$Sex <- as.factor(train$Sex)
#test <- test %>% filter(!is.na(Embarked)) %>% select(-Cabin)
#test$Sex <- as.factor(test$Sex)
#rf_fit <- randomForest(as.factor(Survived) ~ Pclass + Sex + Age + SibSp + Parch + Fare,
#                     data=train,
#                     importance=TRUE,
#                     ntree=2000)
#
# # The importance=TRUE argument allows us
# #to inspect variable importance as we’ll see, and the ntree argument specifies how many trees we want to grow.
#
# varImpPlot(rf_fit)
#
# #There’s two types of importance measures shown above. The accuracy one tests to
# #see how worse the model performs without each variable, so a high decrease in accuracy would be expected
# #for very predictive variables. The Gini one digs into the mathematics behind decision trees, but essentially
# #measures how pure the nodes are at the end of the tree.
# #Again it tests to see the result if each variable is taken out and a high score means the variable was important.
#
# preds_rf <- predict(rf_fit, test)
# caret::confusionMatrix(as.factor(preds_rf), as.factor(test$Survived))
