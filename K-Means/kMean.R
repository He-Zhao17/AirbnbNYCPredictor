rm(list=ls())

colcls=c("Review.Scores.Rating"="factor")
file<-file.choose()
data<- read.csv(file,colClasses=colcls)


kmodel <- kmeans(data[,c(12,7,28,29,30,31,32,35,36,37,5,6,8,9,10)], 2);


kmodel
table(kmodel$cluster, data$Review.Scores.Rating)