.libPaths("D:/Work Space/.libPaths_R Studio")
.libPaths()

df_all = read.csv("F:/Data/creditData.csv", header = T,stringsAsFactors = F)
df_all=df_all[,-1]

set.seed(233)
firstSample = sample(1:nrow(df_all), 1/5*nrow(df_all))
PrivateSet = df_all[firstSample, ]

df_train = df_all[-firstSample, ]

set.seed(1234) 
library(caret)
splitIndex = createDataPartition(df_train$Y,time=1,p=0.7,list=FALSE) 
TrainSet = df_train[splitIndex,] 
TestSet = df_train[-splitIndex,] 

prop.table(table(TrainSet$Y)) 
prop.table(table(TestSet$Y)) 


#===================================================

# TrainSet_Yes = TrainSet[which(TrainSet$Y == 1), ]
# TrainSet_No = TrainSet[which(TrainSet$Y == 0), ]
# nrow(TrainSet_No);nrow(TrainSet_Yes)


Entropy=function(x){
  p=prop.table(table(x))
  entropy=-(sum(p*log2(p)))
  return(entropy)
}

# 构造unlabel数据
set.seed(5)
index = sample(1:nrow(TrainSet), 1/50*nrow(TrainSet))

Data_label = TrainSet[index, ]
Data_unlabel = TrainSet[-index, ]
Data_unlabel$Y = NA

# self-train
library(xgboost)
semi_surpervived = function(Data_label, Data_unlabel){
  accuracy = vector()
  trainData.xgb = xgb.DMatrix(data.matrix(Data_label[,-which(colnames(Data_label) == 'Y')]), 
                              label = Data_label$Y)
  params = list(booster='gbtree',
                eta=0.3,
                max_depth=6,
                lambda=1,
                objective='binary:logistic',
                eval_metric='error')
  xgb = xgb.train(params=params, trainData.xgb, nround=50)
  prob = predict(xgb, data.matrix(Data_unlabel[,-which(colnames(Data_unlabel) == 'Y')]))
  
  pred = ifelse(prob < 0.5, 0, 1)
  qrob = 1-prob
  
  entropy = -(prob*log2(prob)+qrob*log2(qrob))
  index1 = order(entropy, decreasing = F)[1:20]
  newLabel = Data_unlabel[index1,]
  newLabel$Y = pred[index1]
  
  Data_label_N = rbind(Data_label, newLabel)
  Data_unlabel_N = Data_unlabel[-index1, ]
  
  probT = predict(xgb, data.matrix(TestSet[,-which(colnames(TestSet) == 'Y')]))
  predT = ifelse(probT < 0.5, 0, 1)
  accuracy = c(accuracy, mean(TestSet$Y == predT))
  
  return(list(Data_label = Data_label_N, Data_unlabel = Data_unlabel_N, model = xgb, accuracy = accuracy))
}


CruveAccuracy = vector()
while (1) {
  if(nrow(Data_unlabel)<10) {
    finalModel = model$model
    break
  }else{
    model = semi_surpervived(Data_label, Data_unlabel)
    Data_label = model$Data_label
    Data_unlabel = model$Data_unlabel
    CruveAccuracy = c(CruveAccuracy, model$accuracy)
  }
}

# self-train学习进度
x = c(1:length(CruveAccuracy))
splitx = seq(0,1,by=0.01)
indexx = round(length(CruveAccuracy)*splitx)
indexx = indexx[-1]
plot(CruveAccuracy[indexx]~indexx, type='o', lty=2, lwd=2, 
     xlab = 'Iterations', ylab = 'Accuracy', 
     main='Self-Train Learning Cruve')

# finalModel在TestSet上的表现
prob = predict(finalModel, data.matrix(TestSet[,-which(colnames(TestSet) == 'Y')]))
pred = ifelse(prob < 0.5, 0, 1)
table(TestSet$Y, pred)
mean(TestSet$Y != pred)

# finalModel在PrivateSet上的表现
prob = predict(finalModel, data.matrix(PrivateSet[,-which(colnames(PrivateSet) == 'Y')]))
pred = ifelse(prob < 0.5, 0, 1)
table(PrivateSet$Y, pred)
mean(PrivateSet$Y != pred)


