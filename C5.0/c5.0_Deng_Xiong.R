#  Course          : CS513 
#  First Name      : Deng
#  Last Name       : Xiong
#  Id              : 10402740
#  purpose         : Final pj c50

############################## c50 ############################## 

# clear
rm(list=ls())

# Choose the normalized.csv data file
filename <- file.choose()
data<- read.csv(filename,header = TRUE,na.strings = "?")

# No gps info
data <- data[,-c(4:5)]
# summary(data)

# Factor class
data$High.Review.Score <- as.factor(data$High.Review.Score)
data$Room.Type_Entire.home.apt <- as.factor(data$Room.Type_Entire.home.apt)
data$Room.Type_Private.room <- as.factor(data$Room.Type_Private.room)
data$Room.Type_Shared.room <- as.factor(data$Room.Type_Shared.room)
# summary(data)

# Install.packages
library('C50')

# Training and testing
index <- sort(sample(nrow(data), round(.25*nrow(data))))
training <- data[-index,]
testing <- data[index,]

# Summary
C50_list <- C5.0( High.Review.Score~.,data=training[,])
summary(C50_list)
plot(C50_list)
Predict<-predict( C50_list ,testing , type="class" )
table(actual=testing[,31],Predict)

# Error rate
error <- (testing[,31]!=Predict)
errorrate<-sum(error)/length(testing[,31])
accuracy<- 1-errorrate
accuracy

############################## end ############################## 