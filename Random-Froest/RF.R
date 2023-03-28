rm(list=ls())

library(randomForest)
library(class)

#load data
setwd("/Users/pan/Desktop/R/CS513/final project")
df <- read.csv("normalized_nybnb.csv", header = TRUE, na.strings = c("?"))
View(df)
summary(df)

df$High.Review.Score<-factor(df$High.Review.Score, levels = c("0","1"), labels = c("low","high"))
is.factor(df$High.Review.Score)
df$High.Review.Score


View(df)

split_size<-floor(0.70*nrow(df))

random_sample<-sample(seq_len(nrow(df)), size = split_size)

train<-df[random_sample,]
test<-df[-random_sample,]

RF<-randomForest(High.Review.Score~.,train, importance=TRUE, ntree=1000)
importance(RF)
varImpPlot(RF)

# Prediction
pred_RF<-predict(RF,test,type = "class")
length(pred_RF)
length(test)

#Confusion Matric
confMat_RF<-table(test$High.Review.Score,pred_RF)
print(confMat_RF)

accuracy<-function(x){
  sum(diag(x)/sum(rowSums(x)))*100
}

# Accuracy
accuracy(confMat_RF)

