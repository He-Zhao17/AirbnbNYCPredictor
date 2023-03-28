rm(list=ls())

colcls=c("Review.Scores.Rating"="factor")

file<-file.choose()
data<- read.csv(file,colClasses=colcls)
data$Review.Scores.Rating

index<-sort(sample(nrow( data),round(.30*nrow(data ))))
training<- data[-index,]
test<- data[index,]


library(kknn) 
predict_k1 <- kknn(formula= Review.Scores.Rating~., training , test, k=7,kernel ="rectangular"  )

fit <- fitted(predict_k1)
table(test$Review.Scores.Rating,fit)

right<- ( test$Review.Scores.Rating==fit)
rate<-sum(right)/length(right)
rate


