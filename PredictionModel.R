library(caret)
library(doMC)
library(lubridate)
registerDoMC(cores = 4)

data <- read.csv("data/pml-training.csv", na.string = c("NA", ""))
##str(data)
##summary(data)

#set.seed(141)
inTrain <- createDataPartition(y = data$classe, p = 0.6, list = FALSE)
train.feature <- data[inTrain, -ncol(data)]
train.outcome <- data[inTrain, ncol(data), drop = FALSE]
test.feature <- data[-inTrain, -ncol(data)]
test.outcome <- data[-inTrain, ncol(data), drop = FALSE]

train.feature <- train.feature[, -c(1, 2, 4, 5)]
time.feature <- function(data) {
  data[, 1] <- as.POSIXct(data[, 1], origin = "1970-01-01", tz = "UTC")
  data$wday <- wday(data[, 1])
  data$hour <- hour(data[, 1])
  data$minute <- minute(data[, 1])
  data[, -1]
}
train.feature <- time.feature(train.feature)

filter.feature <- function(data, threshold = .95) {
  p.num <- function(x) {
    sum(is.na(x)) / length(x)
  }
  illness <- apply(data, 2, p.num)
  which(illness >= threshold)
}
ill.feature <- filter.feature(train.feature)
train.feature <- train.feature[, -ill.feature]

train.feature[, 1] <- as.numeric(train.feature[, 1])

pca <- preProcess(train.feature, method = "pca", thresh = .975)
train.feature <- predict(pca, train.feature)

redo <- function(data, ill.feature, pca) {
  data <- data[, -c(1, 2, 4, 5)]
  data <- time.feature(data)
  data <- data[, -ill.feature]
  data[, 1] <- as.numeric(data[, 1])
  predict(pca, data)
}

cvControl <- trainControl(method = "cv", number = 5)
#set.seed(592)
svmModel <- train(train.feature, train.outcome$classe, method = "svmRadial", trControl = cvControl)

bootControl <- trainControl(number = 2)
rfModel <- train(train.feature, train.outcome$classe, method = "rf", trControl = bootControl)

test.feature <- redo(test.feature, ill.feature, pca)

svmModel
svmPred <- predict(svmModel, test.feature)
confusionMatrix(svmPred, test.outcome$classe)

rfModel
rfPred <- predict(rfModel, test.feature)
confusionMatrix(rfPred, test.outcome$classe)

