library(caret)
library(doMC)
library(lubridate)
registerDoMC(cores = 4)
sessionInfo()
data <- read.csv("data/pml-training.csv", na.string = c("NA", ""))
set.seed(141)
inTrain <- createDataPartition(y = data$classe, p = 0.6, list = FALSE)
train.feature <- data[inTrain, -ncol(data)]
train.outcome <- data[inTrain, ncol(data), drop = FALSE]
test.feature <- data[-inTrain, -ncol(data)]
test.outcome <- data[-inTrain, ncol(data), drop = FALSE]
train.feature <- train.feature[, -c(1, 2, 5)]
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
train.feature[, 2] <- as.numeric(train.feature[, 2])
num.feature <- ncol(train.feature)
pca <- preProcess(train.feature, method = "pca", thresh = .975)
pca
train.feature <- predict(pca, train.feature)
redo.process <- function(data, ill.feature, pca) {
  data <- data[, -c(1, 2, 5)]
  data <- time.feature(data)
  data <- data[, -ill.feature]
  data[, 2] <- as.numeric(data[, 2])
  predict(pca, data)
}
ptm <- proc.time()
set.seed(592)
cv.control <- trainControl(method = "cv", number = 5)
svm.model <- train(train.feature, train.outcome$classe, method = "svmRadial", trControl = cv.control, tuneLength = 5)
svm.running.time <- proc.time() - ptm
ptm <- proc.time()
set.seed(653)
boot.control <- trainControl(number = 5)
rf.model <- train(train.feature, train.outcome$classe, method = "rf", trControl = boot.control)
rf.running.time <- proc.time() - ptm
test.feature <- redo.process(test.feature, ill.feature, pca)
svm.running.time
svm.model
svm.pred <- predict(svm.model, test.feature)
confusionMatrix(svm.pred, test.outcome$classe)
rf.running.time
rf.model
rf.pred <- predict(rf.model, test.feature)
confusionMatrix(rf.pred, test.outcome$classe)
test <- read.csv("data/pml-testing.csv", na.string = c("NA", ""))
test <- test[, -ncol(test)]
test <- redo.process(test, ill.feature, pca)
predict(svm.model, test)
predict(rf.model, test)