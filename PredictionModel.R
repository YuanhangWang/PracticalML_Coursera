data <- read.csv("data/pml-training.csv")
str(data)
summary(data)
data <- data[, -2]

library(caret)
set.seed(141)
inTrain <- createDataPartition(y = data$classe, p = 0.6, list = FALSE)
train <- data[inTrain, ]
test <- data[-inTrain, ]
rm(data)

filter.feature <- function(data, threshold = .95) {
  p.num <- function(x) {
    sum(is.na(x)) / length(x)
  }
  illness <- apply(data, 2, p.num)
  well.conditioned <- illness[illness < threshold]
  names(well.conditioned)
}
well.conditioned.feature <- filter.feature(train)
train <- train[, well.conditioned.feature]
test <- test[, well.conditioned.feature]

cvControl <- trainControl(method = "cv")
set.seed(592)
svmModel <- train(classe ~ ., data = train, method = "svmRadial", trControl = cvControl, scaled = TRUE)

set.seed(653)
rfModel <- train(classe ~ ., data = train, method = "rf")
