rm(list=ls())

library(e1071)
library(class)


#load data
setwd("/Users/pan/Desktop/R/CS513/final project")
df <- read.csv("normalized_nybnb.csv", header = TRUE, na.strings = c("?"))
View(df)
summary(df)

df$High.Review.Score<-factor(df$High.Review.Score, levels = c("0","1"), labels = c("low","high"))
is.factor(df$High.Review.Score)
df$High.Review.Score

sampleSize <- floor(0.70 * nrow(df))


set.seed(100)
train <- sample(seq_len(nrow(df)), size = sampleSize)

trainingData <- df[train, ]
testData <- df[-train, ]

NB<- naiveBayes(High.Review.Score ~ ., data = trainingData)

predictNB <- predict(NB, testData)

conf_matrix <- table(predictNB=predictNB,class=testData$High.Review.Score)
print(conf_matrix)


#Accuracy and error rating
accuracy <- function(x){sum(diag(x)/(sum(rowSums(x)))) * 100}
accuracy(conf_matrix)
