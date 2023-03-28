#  Course          : CS513 
#  First Name      : Deng
#  Last Name       : Xiong
#  Id              : 10402740
#  purpose         : Final pj cart

############################## cart ############################## 

rm(list=ls())
#install.packages("rpart")
#install.packages("rpart.plot")     
#install.packages("rattle")         
#install.packages("RColorBrewer") 

# Load Packages
library(rpart)
library(rpart.plot)		
library(rattle)											
library(RColorBrewer)

# Choose the normalized.csv data file
filename<-file.choose()
data<-read.csv(filename)
# View(data)

# No gps info
data <- data[,-c(4:5)]

# factor class
data$Room.Type_Entire.home.apt <- as.factor(data$Room.Type_Entire.home.apt)
data$Room.Type_Private.room <- as.factor(data$Room.Type_Private.room)
data$Room.Type_Shared.room <- as.factor(data$Room.Type_Shared.room)
data$High.Review.Score <- as.factor(data$High.Review.Score)
# data <- na.omit(data)
# View(data)

# Training and testing
index <- sort(sample(nrow(data),as.integer(.3*nrow(data))))
training<- data[-index,]
testing<- data[index, ]
summary(testing)
summary(training)

# CART 
Dtree<-rpart(High.Review.Score~.,data=training)
fancyRpartPlot(Dtree)

# Prediction
prediction<-predict(Dtree	,testing,	type="class")
table(Actual=testing[,31],prediction)

# Error rate of DT
right<-(testing[,31]==prediction)
accuracy<-sum(right)/length(right)
accuracy

############################## end ############################## 






 