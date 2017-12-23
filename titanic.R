setwd("H:/Data Science/Kaggle/Titanic/")
library(ggplot2)
library(stringr)
answer <- read.csv("answer.csv")
train <- read.csv("train.csv", na.strings = c("NA","","NaN"))
test <- read.csv("test.csv", na.strings = c("NA","","NaN"))
test$Survived <- NA
titanic <- rbind(train,test)
titanic$Survived <- as.factor(titanic$Survived)
rm(test,train)

head(titanic)
str(titanic)

ggplot(titanic, aes(Title,Age)) + geom_boxplot()

## Feature Engineering

#Title

titanic$Name <- as.character(titanic$Name)
# names <- gsub(pattern = "[[:punct:]]+", replacement = " ",x = titanic$Name)
# names <- gsub(pattern = "  ", replacement = " ",x = names)
# names <- gsub(pattern = " $", replacement = " ",x = names)
# VIP <- c("Capt","Col","Don","Dona","Dr","Jonkheer","Lady","Major",
#          "Mlle", "Mme","Rev","Sir","the Countess")

titanic$Title <- gsub('(.*, )|(\\..*)', '', titanic$Name)
# (.*, ) remove all chars before a comma+space and (\\..*) removes all chars after a point(.)

titanic$Title <- gsub(pattern = "Ms",replacement = "Miss",x = titanic$Title)
titanic$Title <- gsub(pattern = "Mme|Mlle",replacement = "Mlle",x = titanic$Title)
titanic$Title <- gsub(pattern = "Jonkheer|Don|Sir|the Countess|Dona|Lady",replacement = "Royalty",x = titanic$Title)
titanic$Title <- gsub(pattern = "Capt|Col|Major|Dr|Rev",replacement = "Officer",x = titanic$Title)


#Class

titanic$Pclass <- factor(titanic$Pclass,labels = c("Upper", "Middle", "Lower"))

#Ticket
titanic$Ticket <- word(gsub("[[:punct:]]+","",titanic$Ticket),1)
titanic$Ticket <- ifelse(grepl(pattern = "^[[:digit:]]",
                             x= titanic$Ticket),
                       yes = "XXX",
                       no = titanic$Ticket)

#FamilySize
titanic$Family <- titanic$SibSp + titanic$Parch + 1

#Singleton
titanic$Singleton <- ifelse(titanic$Family==1,"Y","N")

#Small Family
titanic$SmallFamily <- ifelse(titanic$Family<3,"Y","N")

#Big Family
titanic$BigFamily <- ifelse(titanic$Family>2, "Y","N")

#FamilyID

titanic$Surname <- gsub(pattern = ",.*",replacement = "",x = titanic$Name)
titanic$FamilyID <- paste(as.character(titanic$Family),titanic$Surname,sep = "")
famIDs <- data.frame(table(titanic$FamilyID))
famIDs <- famIDs[famIDs$Freq <= 3,]
titanic$FamilyID[titanic$FamilyID %in% famIDs$Var1] <- "Small"
rm(famIDs)
## Missing Values

#Embarked

sum(is.na(titanic$Embarked)) # 2 missing values
table(titanic$Embarked) # since most of the values are 'S', Let's Replace the missing values with S
titanic$Embarked[is.na(titanic$Embarked)] <- "S"

levels(titanic$Embarked) <- c("Cherbourg","Queenstown","Southampton")

#Cabin status
#For now, let's assume that rather than cabin number, the availability of cabin is more important.

# titanic$CabinStatus <- ifelse(is.na(titanic$Cabin),"N","Y") 

titanic$Cabin <- substr(gsub("[^[:alpha:]]+","",titanic$Cabin),1,1)
titanic$Cabin <- ifelse(is.na(titanic$Cabin),"NONE",titanic$Cabin)

# Fare
titanic$Fare[is.na(titanic$Fare)] <- median(titanic$Fare,na.rm = TRUE)

# Age
library(rpart)
Agefit <- rpart(Age ~ Pclass + Sex + SibSp + Parch + Fare + Embarked + Title + Family,
                data=titanic[!is.na(titanic$Age),],
                method = "anova")
titanic$Age[is.na(titanic$Age)] <- predict(Agefit, titanic[is.na(titanic$Age),])

str(titanic)
charvars <- c("Ticket","Cabin","Title","Singleton","SmallFamily",
              "BigFamily","Surname","FamilyID")
titanic[,charvars] <- lapply(titanic[,charvars],as.factor)
rm(Agefit,charvars)
# 
# library(missForest)
# titanic_impute <- titanic[,-c(1,2,4,7,8,11)]
# 
# set.seed(555)
# trainmis <- missForest(titanic_impute,
#                        verbose = FALSE,
#                        ntree = 500,
#                        mtry = 3,
#                        variablewise = TRUE)

# m <- c(1:6)
# tree <- c(50,150,250,350,500)
# errortable <- NULL
# for(i in m){
#         for(j in tree){
#                 set.seed(555)
#                 trainmis <-missForest(titanic_impute,
#                                       verbose = FALSE,
#                                       ntree = j,
#                                       mtry = i,
#                                       variablewise = TRUE)
#                 errortable <- rbind(errortable,cbind(t(trainmis$OOBerror),i,j))
#         }
# }

# titanic_impute <- trainmis$ximp
# rm(trainmis,i,j,m,tree)
# 
# imputed_combined <- data.frame(titanic_impute,Survived = titanic$Survived)

## Model training
titanic_clean <- titanic[!is.na(titanic$Survived),]
test_clean <- titanic[is.na(titanic$Survived),]
set.seed(555)
ind <- sample(x = 1:nrow(titanic_clean),size = round(0.8*nrow(titanic_clean)),replace = FALSE)
train <- titanic_clean[ind,]
test <- titanic_clean[-ind,]
rm(ind)

library(gbm)
train <- titanic_clean[,c(predictors,response)]
test <- test_clean[,predictors]

tree <- c(150,250,500,700,1000)
rate <- seq(0.05,0.2,0.01)
depth <- c(1,2,3,4)
fraction <- seq(0.1,1,0.1)
nodes <- 1:5
accuracytable <- NULL
train$Survived <- as.numeric(as.character(train$Survived))
for(i in tree){
        for(j in rate){
                for(k in depth){
                        for(l in fraction){
                                for(m in nodes){
                                        set.seed(1)
                                        boost.model <- gbm(Survived~.,
                                                           data= train,
                                                           distribution = "bernoulli",
                                                           n.trees=i,
                                                           interaction.depth = k,
                                                           shrinkage = j,
                                                           bag.fraction = l,
                                                           n.minobsinnode = m)
                                        boost.predict <- round(predict(boost.model,
                                                                 newdata = test,
                                                                 n.trees = i,
                                                                 type='response'))
                                        accuracy <- sum(answer$Survived==boost.predict)/nrow(test)
                                        accuracytable <- rbind(accuracytable,cbind(accuracy,i,j,k,l,m))
                                }

                        }

                }
        }
}
rm(accuracy,i,j,k,l,m,nodes,fraction,depth,rate,tree,boost.model,boost.predict)

set.seed(555)
gbmmodel <- gbm(Survived~.,
                data = train,
                distribution = "bernoulli",
                n.trees = 150,
                interaction.depth = 4,
                shrinkage = 0.2,
                bag.fraction = 0.5,
                n.minobsinnode = 5)
predict <- round(predict(gbmmodel,
                         newdata = test,
                         n.trees = 150,
                         type = "response"))
accuracy <- sum(test$Survived==predict)/nrow(test)

## PREDICTION
Survived <- round(predict(gbmmodel,
                          newdata = test_clean,
                          n.trees = 150,
                          type = "response"))
PassengerId <- titanic$PassengerId[is.na(titanic$Survived=="NA")]
submission <- data.frame(PassengerId,Survived)
write.csv(x = submission,file = "submission.csv",row.names = FALSE)
# seed 555, gbm n.trees 150, inter.depth 3, shrinkage 0.12, bag.fraction 0.5, n.minnode 5. 


#Random Forest training
library(randomForest)

# m <- c(1:9)
# tree <- c(50,150,250,350,500)
# accuracytable <- NULL
# for(i in m){
#         for(j in tree){
#                 set.seed(555)
#                 model <-randomForest(Survived~. ,
#                                      data = train,
#                                      ntree = j,
#                                      mtry = i,
#                                      variablewise = TRUE)
#                 predict <- predict(model,
#                                    newdata = test,
#                                    type='response')
#                 accuracy <- sum(test$Survived==predict)/nrow(test)
#                 accuracytable <- rbind(accuracytable,cbind(accuracy,i,j))
#         }
# }

rfmodel <- randomForest(Survived ~.,
                        data = train,
                        ntree = 150,
                        mtry = 3)

rf_Survived <- predict(rfmodel,
                       newdata = test_clean,
                       type = "response")
PassengerId <- titanic$PassengerId[is.na(titanic$Survived=="NA")]
submission <- data.frame(PassengerId,rf_Survived)
write.csv(x = submission,file = "rfsubmission.csv",row.names = FALSE)


#Cforest Modelling
library(party)
set.seed(415)
partymodel <- cforest(Survived~Pclass + Sex + Age + SibSp + Parch + Fare +
                              Embarked + Title + Family + FamilyID + Cabin+
                              Singleton + SmallFamily + BigFamily + Surname +
                              Ticket,
                      data = titanic_clean,
                      controls = cforest_unbiased(ntree = 2000,mtry =3))
Prediction <- predict(partymodel, test_clean, OOB=TRUE, type = "response")
submission <- data.frame(PassengerId = test_clean$PassengerId,Survived = Prediction)
write.csv(x = submission,file = "partysubmission.csv",row.names = FALSE)

#rf new
library(randomForest)
set.seed(555)
rfmodel <- randomForest(Survived ~Pclass + Sex + Age + SibSp + Parch + Fare +
                                Embarked + Title + Family + FamilyID + Cabin + Ticket,
                        data = titanic_clean,
                        ntree = 2000,
                        mtry = 4)

rf_Survived <- predict(rfmodel,
                       newdata = test_clean,
                       type = "response")
submission <- data.frame(PassengerId = test_clean$PassengerId,Survived = rf_Survived)
write.csv(x = submission,file = "rfsubmission.csv",row.names = FALSE)


#Ensemble Method
library(caret)
myControl <- trainControl(method = "cv",
                          number = 5,
                          returnResamp = "none"
                          # savePredictions = 'final',
                          # classProbs = TRUE
                          )

predictors <- c("Pclass","Sex","Age","SibSp","Parch","Fare","Embarked",
                "Title","Family","FamilyID")

response <- "Survived"

levels(titanic_clean$Survived) <- c("N","Y")
set.seed(1)
cforest <- train(titanic_clean[,predictors],titanic_clean[,response],
                 method = 'cforest',
                 trControl = myControl)

set.seed(1)
rf <- train(titanic_clean[,predictors],titanic_clean[,response],
            method = 'rf',
            trControl = myControl)

set.seed(1)
gbm <- train(titanic_clean[,predictors],titanic_clean[,response],
             method = 'gbm',
             trControl = myControl)
# glm <- train(titanic_clean[,predictors],titanic_clean[,response],
#              method = 'glm',
#              trControl = myControl)

set.seed(1)
rpart <- train(titanic_clean[,predictors],titanic_clean[,response],
               method = 'rpart',
               trControl = myControl)

set.seed(1)
nnet <- train(titanic_clean[,predictors],titanic_clean[,response],
              method = 'nnet',
              trControl = myControl)

set.seed(1)
ada <- train(titanic_clean[,predictors],titanic_clean[,response],
             method = 'ada',
             trControl = myControl)

set.seed(1)
treebag <- train(titanic_clean[,predictors],titanic_clean[,response],
             method = 'treebag',
             trControl = myControl)

set.seed(1)
adaboost <- train(titanic_clean[,predictors],titanic_clean[,response],
                  method = 'adaboost',
                  trControl = myControl)

set.seed(1)
adaboostm1 <- train(titanic_clean[,predictors],titanic_clean[,response],
                  method = 'AdaBoost.M1',
                  trControl = myControl)

set.seed(1)
c5 <- train(titanic_clean[,predictors],titanic_clean[,response],
                  method = 'C5.0',
                  trControl = myControl)

set.seed(1)
rpart2 <- train(titanic_clean[,predictors],titanic_clean[,response],
                  method = 'rpart2',
                  trControl = myControl)

set.seed(1)
ctree <- train(titanic_clean[,predictors],titanic_clean[,response],
                  method = 'ctree2',
                  trControl = myControl)

# set.seed(400)
# xgb <- train(titanic_clean[,predictors],titanic_clean[,response],
#                   method = 'xgbTree',
#                   trControl = myControl)

      
#Predicting the out of fold prediction probabilities for training data
titanic_clean$OOF_cforest <- cforest$pred$Y[order(cforest$pred$rowIndex)]
titanic_clean$OOF_rf <- rf$pred$Y[order(rf$pred$rowIndex)]
titanic_clean$OOF_gbm <- gbm$pred$Y[order(gbm$pred$rowIndex)]
titanic_clean$OOF_rpart <- rpart$pred$Y[order(rpart$pred$rowIndex)]
titanic_clean$OOF_nnet <- nnet$pred$Y[order(nnet$pred$rowIndex)]
titanic_clean$OOF_ada <- ada$pred$Y[order(ada$pred$rowIndex)]
titanic_clean$OOF_treebag <- treebag$pred$Y[order(treebag$pred$rowIndex)]
titanic_clean$OOF_adaboost <- adaboost$pred$Y[order(adaboost$pred$rowIndex)]
titanic_clean$OOF_adaboostm1 <- adaboostm1$pred$Y[order(adaboostm1$pred$rowIndex)]
titanic_clean$OOF_c5 <- c5$pred$Y[order(c5$pred$rowIndex)]
titanic_clean$OOF_rpart2 <- rpart2$pred$Y[order(rpart2$pred$rowIndex)]
titanic_clean$OOF_ctree <- ctree$pred$Y[order(ctree$pred$rowIndex)]


#Predicting probabilities for the test data
test_clean$OOF_cforest <- predict(cforest, test_clean[,predictors], type = 'prob')$Y
test_clean$OOF_rf <- predict(rf, test_clean[,predictors], type = 'prob')$Y
test_clean$OOF_gbm <- predict(gbm, test_clean[,predictors], type = 'prob')$Y
test_clean$OOF_rpart <- predict(rpart, test_clean[,predictors], type = 'prob')$Y
test_clean$OOF_nnet <- predict(nnet, test_clean[,predictors], type = 'prob')$Y
test_clean$OOF_ada <- predict(ada, test_clean[,predictors], type = 'prob')$Y
test_clean$OOF_treebag <- predict(treebag, test_clean[,predictors], type = 'prob')$Y
test_clean$OOF_adaboost <- predict(adaboost, test_clean[,predictors], type = 'prob')$Y
test_clean$OOF_adaboostm1 <- predict(adaboostm1, test_clean[,predictors], type = 'prob')$Y
test_clean$OOF_c5 <- predict(c5, test_clean[,predictors], type = 'prob')$Y
test_clean$OOF_rpart2 <- predict(rpart2, test_clean[,predictors], type = 'prob')$Y
test_clean$OOF_ctree <- predict(ctree, test_clean[,predictors], type = 'prob')$Y

## Top Layer with xgb
predtop <- c("OOF_cforest","OOF_rf","OOF_gbm","OOF_rpart","OOF_nnet","OOF_ada","OOF_treebag",
             "OOF_adaboost","OOF_adaboostm1","OOF_c5","OOF_rpart2","OOF_ctree")
set.seed(400)
xgbtop <- train(titanic_clean[,predtop],titanic_clean[,response],
                method = 'xgbTree',
                trControl = myControl)

Survived <- predict(xgbtop,test_clean[,predtop])
levels(Survived) <- c(0,1)

predict1 <- predict(cforest,test_clean[,predictors])
predict2 <- predict(rf,test_clean[,predictors])
predict3 <- predict(gbm,test_clean[,predictors])
predict4 <- predict(rpart,test_clean[,predictors])
predict5 <- predict(nnet,test_clean[,predictors])
predict6 <- predict(ada,test_clean[,predictors])
predict7 <- predict(treebag,test_clean[,predictors])
predict8 <- predict(adaboost,test_clean[,predictors])
predict9 <- predict(adaboostm1,test_clean[,predictors])
predict10 <- predict(c5,test_clean[,predictors])
predict11 <- predict(rpart2,test_clean[,predictors])
predict12 <- predict(ctree,test_clean[,predictors])
# 
# predict <- data.frame(predict1,predict2,predict3,predict4,predict5,predict6,predict7,predict8,predict9,
#                       predict10,predict11,predict12)
# predict[,1:12] <- lapply(predict[,1:12],as.character)
# predict[,1:12] <- lapply(predict[,1:12],as.numeric)
# predict$sum <- rowSums(predict)
# 
# Survived <- ifelse(predict$sum >= 6, 1,0)

submission <- data.frame(PassengerId = test_clean$PassengerId,Survived)
write.csv(x = submission,file = "ensemble12xgbtop.csv",row.names = FALSE)


# XTreme Gradient BOosting
library(xgboost)

predictors <- c("Pclass","Sex","Age","SibSp","Parch","Fare","Embarked",
                "Title","Family","FamilyID", "Cabin","Ticket")

response <- "Survived"
library(Matrix)
sparse_matrix <- sparse.model.matrix(Survived~.-1,data = titanic_clean[,c(predictors,response)])
sparse_test <- sparse.model.matrix(object = ~.-1,data = test_clean[,predictors])
y <- as.numeric(as.character(titanic_clean$Survived))

xgb <- xgboost(data = data.matrix(sparse_matrix),
               label = y,
               eta = 0.1,
               max_depth = 15, 
               nround=25, 
               subsample = 0.5,
               colsample_bytree = 0.5,
               seed = 400,
               eval_metric = "error",
               objective = "binary:logistic"
               )


#
# tree <- c(100,150,250,500)
# rate <- seq(0.05,0.2,0.01)
# depth <- c(1,2,3,4)
# fraction <- seq(0.1,1,0.1)
# nodes <- 1:5
# accuracytable <- NULL
# for(i in tree){
#         for(j in rate){
#                 for(k in depth){
#                         for(l in fraction){
#                                 for(m in nodes){
#                                         set.seed(555)
#                                         boost.model <- xgboost(data = data.matrix(sparse_matrix),
#                                                                label = y,
#                                                                eta = 0.1,
#                                                                max_depth = 15,
#                                                                nround=25,
#                                                                subsample = 0.5,
#                                                                colsample_bytree = 0.5,
#                                                                seed = 400,
#                                                                eval_metric = "error",
#                                                                objective = "binary:logistic"
#                                         )
#                                         boost.predict <- round(predict(boost.model,
#                                                                        newdata = test,
#                                                                        n.trees = i,
#                                                                        type='response'))
#                                         accuracy <- sum(test$Survived==boost.predict)/nrow(test)
#                                         accuracytable <- rbind(accuracytable,cbind(accuracy,i,j,k,l,m))
#                                 }
#
#                         }
#
#                 }
#         }
# }
# rm(accuracy,i,j,k,l,m,nodes,fraction,depth,rate,tree,boost.model,boost.predict)


Survived <- round(predict(object = xgb,newdata = as.matrix(sparse_test)))

library(caret)
myControl <- trainControl(method = "cv",
                          number = 5,
                          # returnResamp = "none"
                          savePredictions = 'final',
                          classProbs = TRUE
)
y <- factor(y,labels = c("N","Y"))

set.seed(400)
glm <- train(data.matrix(sparse_matrix),as.factor(y),
             method = "glm",
             trControl = myControl)

set.seed(1)
cforest <- train(data.matrix(sparse_matrix),as.factor(y),
               method = "cforest",
               trControl = myControl)

set.seed(400)
xgb <- train(data.matrix(sparse_matrix),as.factor(y),
                 method = "xgbTree",
                 trControl = myControl)


Survived1 <- predict(object = glm,newdata = as.matrix(sparse_test), type = 'prob')$Y
Survived2 <- predict(object = cforest,newdata = as.matrix(sparse_test), type = 'prob')$Y
Survived3 <- predict(object = xgb,newdata = as.matrix(sparse_test), type = 'prob')$Y

Survived <- round((Survived1+Survived2+Survived3)/3)

Survived <- predict(object = cforest, newdata = as.matrix(sparse_test))

# Result
sum(Survived==answer$Survived)/nrow(answer)




## gbm

fitControl <- trainControl(method = 'repeatedcv',
                           number = 3,
                           repeats = 3)

grid <- expand.grid(n.trees = c(50,100,200,300),
                    interaction.depth = c(4:12),
                    shrinkage = seq(0.001,0.101,0.005),
                    n.minobsinnode = 10)

gbmfit <- train(Survived ~. , 
                data = titanic_clean[c(predictors,response)],
                trcontrol = fitControl,
                bag.fraction = 0.5,
                verbose = FALSE,
                set.seed = 1234)
