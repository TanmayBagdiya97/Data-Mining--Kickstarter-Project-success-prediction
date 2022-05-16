#load libraries
library(tidyverse)
library(lubridate)
library(text2vec)
library(tm)
library(tidyr)
library(readr)
library(tree)
library(xgboost)
library(rpart)
library(caret)
library(randomForest)

#load data files
train_x <- read_csv("ks_training_X.csv")
train_y <- read_csv("ks_training_y.csv")
test_x <- read_csv("ks_test_X.csv")

## Binding testing and training dataset. 
## Will separate it before training the model
## Makes it easier to data preprocess and avoids discrepancies
combined_data<-rbind(train_x,test_x)



remove_zero<- function(df){
  df[seq_along(df)%% 2==1]
}


## Data Pre Processing
## Imputing Na values
## Converting string column to factors
## Feature Engineering

combined_data <- combined_data %>%
  left_join(train_y, by = "id") %>%
  mutate(avg_reward_amt=unlist(lapply(lapply(lapply(stringr::str_extract_all(reward_amounts, "\\d+"),remove_zero),as.numeric),mean)),
         days_active=difftime(deadline,launched_at,units='days'),
         launch_month=month(launched_at)) %>%
  group_by(category_name) %>%
  mutate(count=n())%>%
  ungroup()%>%
  mutate(category_name=if_else(count<20,'Other',category_name))%>%
  mutate(location_slug =if_else(location_slug=="None","US",str_sub(location_slug, start= -2)))%>%
  mutate(smiling_project=if_else(smiling_project>100,100,round(smiling_project,0)),
         smiling_creator=if_else(smiling_creator>100,100,round(smiling_creator,0)))%>%
  mutate(isbwImg1=ifelse (is.na(isbwImg1),2,isbwImg1),
         color_foreground=ifelse (is.na(color_foreground),'NULL',color_foreground),
         color_background=ifelse (is.na(color_background),'NULL',color_background),
         isTextPic=ifelse (is.na(isTextPic),'NULL',isTextPic),
         isLogoPic=ifelse (is.na(isLogoPic),'NULL',isLogoPic),
         isCalendarPic=ifelse (is.na(isCalendarPic),'NULL',isCalendarPic),
         isDiagramPic=ifelse (is.na(isDiagramPic),'NULL',isDiagramPic),
         isShapePic=ifelse (is.na(isShapePic),'NULL',isShapePic)
         
  )%>%
  mutate(afinn_pos_norm=round(afinn_pos/num_words,4),
         afinn_neg_norm=round(afinn_neg/num_words,4))%>%
  mutate(success = as.factor(success),
         isTextPic = as.factor(isTextPic),
         isLogoPic = as.factor(isLogoPic),
         launch_month= as.factor(launch_month),
         isCalendarPic = as.factor(isCalendarPic),
         isDiagramPic = as.factor(isDiagramPic),
         isShapePic=as.factor(isShapePic),
         days_active=as.numeric(days_active),
         afinn_pos_norm=ifelse (is.na(afinn_pos_norm),0,afinn_pos_norm),
         afinn_neg_norm=ifelse (is.na(afinn_neg_norm),0,afinn_neg_norm),
         avg_reward_amt=ifelse (is.na(avg_reward_amt),0,avg_reward_amt),
         perday= goal/days_active,
         ADV=ifelse (is.na(round(ADV/num_words,4)),0,round(ADV/num_words,4)),
         NOUN=ifelse (is.na(round(NOUN/num_words,4)),0,round(NOUN/num_words,4)),
         ADP=ifelse (is.na(round(ADP/num_words,4)),0,round(ADP/num_words,4)),
         PRT=ifelse (is.na(round(PRT/num_words,4)),0,round(PRT/num_words,4)),
         DET=ifelse (is.na(round(DET/num_words,4)),0,round(DET/num_words,4)),
         PRON=ifelse (is.na(round(PRON/num_words,4)),0,round(PRON/num_words,4)),
         VERB=ifelse (is.na(round(VERB/num_words,4)),0,round(VERB/num_words,4)),
         NUM=ifelse (is.na(round(NUM/num_words,4)),0,round(NUM/num_words,4)),
         CONJ=ifelse (is.na(round(CONJ/num_words,4)),0,round(CONJ/num_words,4)),
         ADJ=ifelse (is.na(round(ADJ/num_words,4)),0,round(ADJ/num_words,4)))

## Removing columns which are not needed
combined_data <- combined_data%>%select(-c(id,big_hit,creator_id, backers_count,creator_name,captions,name,blurb,tag_names,created_at,accent_color,isbwImg1,accent_color,color_foreground,color_background,avg_wordlengths,sentence_counter,avgsentencelength,avgsyls,reward_descriptions,deadline,launched_at,count,reward_amounts,afinn_pos,afinn_neg))

## We tried normalization, text mining and outlier treatment
## But we got either bad accuracy or coding errors

#prep_fun = tolower
#cleaning_tokenizer <- function(v) {
#  v %>%
#    removeNumbers %>% #remove all numbers
#    removePunctuation %>%
#    removeWords(stopwords(kind="en")) %>% #remove stopwords
#    stemDocument %>%
#    word_tokenizer 
#}
#tok_fun = cleaning_tokenizer
#
## Iterate over the individual documents and convert them to tokens
## Uses the functions defined above.
#it_train = itoken(train$blurb, 
#                  preprocessor = prep_fun, 
#                  tokenizer = tok_fun, 
#                  ids = train$id, 
#                  progressbar = FALSE)
#
## Create the vocabulary from the itoken object
#vocab = create_vocabulary(it_train)
#vocab_small = prune_vocabulary(vocab, vocab_term_max = 500)
#
## Create a vectorizer object using the vocabulary we learned
#vectorizer = vocab_vectorizer(vocab_small)
#
## Convert the training documents into a DTM and make it a binary BOW matrix
#dtm_train = create_dtm(it_train, vectorizer)
#dim(dtm_train)
#dtm_train_bin <- dtm_train>0+0


# Convert the small sparse matrix into a dense one
#dense = as.matrix(dtm_train_bin)+0
#
#dense = subset(dense, select=-c(get("goal")))
#
#dense = subset(dense, select=-c(get("success")))
## Use cbind() to append the columns
#train <- cbind(train, dense)
#
#goal=normalize(goal),
#numfaces_project=normalize(numfaces_project),
#numfaces_creator=normalize(numfaces_creator),
#male_project=normalize(male_project),
#male_creator=normalize(male_creator),
#female_project=normalize(female_project),
#female_creator=normalize(female_creator),
#smiling_project=normalize(smiling_project),
#smiling_creator=normalize(smiling_creator),
#minage_project=normalize(minage_project),
#minage_creator=normalize(minage_creator),
#maxage_project=normalize(maxage_project),
#maxage_creator=normalize(maxage_creator),
#num_words=normalize(num_words),
#grade_level=normalize(grade_level),
#avg_reward_amt=normalize(avg_reward_amt),
#days_active=normalize(days_active),
#perday=normalize(perday),
#avg_wordlengths=normalize(avg_wordlengths),
#sentence_counter=normalize(sentence_counter),
#avgsentencelength=normalize(avgsentencelength),
#avgsyls=normalize(avgsyls),
#goal=ifelse(goal>3,3,goal),
#numfaces_project=ifelse(numfaces_project>3,3,numfaces_project),
#numfaces_creator=ifelse(numfaces_creator>3,3,numfaces_creator),
#male_project=ifelse(male_project>3,3,male_project),
#male_creator=ifelse(male_creator>3,3,male_creator),
#female_project=ifelse(female_project>3,3,female_project),
#female_creator=ifelse(female_creator>3,3,female_creator),
#smiling_project=ifelse(smiling_project>3,3,smiling_project),
#smiling_creator=ifelse(smiling_creator>3,3,smiling_creator),
#minage_project=ifelse(minage_project>3,3,minage_project),
#minage_creator=ifelse(minage_creator>3,3,minage_creator),
#maxage_project=ifelse(maxage_project>3,3,maxage_project),
#maxage_creator=ifelse(maxage_creator>3,3,maxage_creator),
#num_words=ifelse(num_words>3,3,num_words),
#grade_level=ifelse(grade_level>3,3,grade_level),
#avg_reward_amt=ifelse(avg_reward_amt>3,3,avg_reward_amt),
#days_active=ifelse(days_active>3,3,days_active),
#perday=ifelse(perday>3,3,perday),
#avg_wordlengths=ifelse(avg_wordlengths>3,3,avg_wordlengths),
#sentence_counter=ifelse(sentence_counter>3,3,sentence_counter),
#avgsentencelength=ifelse(avgsentencelength>3,3,avgsentencelength),
#avgsyls=ifelse(avgsyls>3,3,avgsyls)



## Splitting training and testing data back after preprocessing
train<-combined_data[1:nrow(train_x),]
test<-combined_data[(nrow(train_x)+1):nrow(combined_data),]


## Splitting training data into training and validation(70:30) 
train_inst = sample(nrow(train), .7*nrow(train))

data_train <- train[train_inst,]
data_valid <- train[-train_inst,]

## PREDICTING FOR Y = SUCCESS

## Attempt 1: Logistic Regression


logistic_success <- glm(success~., data = data_train, family = "binomial")
summary(logistic_success)
probs_success_train <- predict(logistic_success, newdata = data_train%>%select(-success), type = "response")
probs_success_valid <- predict(logistic_success, newdata = data_valid%>%select(-success), type = "response")


#make binary classifications training
classifications_success_train <- ifelse(probs_success_train > 0.45, "YES", "NO")
classifications_success_train <- ifelse(is.na(classifications_success_train), "NO", classifications_success_train)


#make binary classifications validation
classifications_success_valid <- ifelse(probs_success_valid > .45, "YES", "NO")
classifications_success_valid <- ifelse(is.na(classifications_success_valid), "NO", classifications_success_valid)

## Training
classifications<-classifications_success_train
classifications_bin<-ifelse(classifications=='YES',1,0)
actuals<-data_train['success']
actuals_bin<-ifelse(actuals$success=='YES',1,0)


## Validation
classifications<-classifications_success_valid
classifications_bin<-ifelse(classifications=='YES',1,0)
actuals<-data_valid['success']
actuals_bin<-ifelse(actuals$success=='YES',1,0)



CM = table(actuals_bin, classifications_bin)


TP <- CM[2,2]
TN <- CM[1,1]
FP <- CM[1,2]
FN <- CM[2,1]

TPR <- TP/(TP+FN)
TNR <- TN/(TN+FP)
FPR <- 1-TNR

accuracy <- (TP+TN)/(TP+TN+FP+FN)



## Attempt 2: Decision Trees

mycontrol = tree.control(nobs = nrow(train), mincut = 5, minsize = 10, mindev = 0.0001)
full_tree <- tree(success ~ . ,
                  data = data_train,
                  control=mycontrol)


accuracy <- function(classifications, actuals){
  correct_classifications <- ifelse(classifications == actuals, 1, 0)
  acc <- sum(correct_classifications)/length(classifications)
  return(acc)}

## Hyper Parameter tuning 
## Have commented the code because it takes a lot of time to run
## Uncomment before running

#treelength <- c(2, 4, 6, 8, 10, 15, 20, 25, 30, 35,40,100,150,200)
#valid_acc <- rep(0, length(treelength))
#train_acc <- rep(0, length(treelength))
#
#for (i in 1:length(treelength)) {
#  pruned_tree=prune.tree(full_tree, best =treelength[i])
#  tree_pred_train <- predict(pruned_tree,newdata=data_train)[,2]
#  tree_pred_test <- predict(pruned_tree,newdata=data_valid)[,2]
#  tree_pred_train <- ifelse(tree_pred_train>0.5,'YES','NO')
#  tree_pred_test <- ifelse(tree_pred_test>0.5,'YES','NO')
#  train_acc[i] <- accuracy(tree_pred_train,data_train$success)
#  valid_acc[i] <- accuracy(tree_pred_test,data_valid$success)
#}
#
#
#plot(treelength,train_acc,type="l",col="blue",main="Accuracy-Tree Length",
#     xlab="Tree Length", ylab="Accuracy")
#lines(treelength,valid_acc,col="green")
#legend("bottomright",  
#       legend = c("Train", "Valid"),
#       col = c("blue", "green"),
#       lwd=2)
#
### We got the best accuracy for tree length=15
#Best_accuracy<-max(valid_acc)

## Training on tree length=15
pruned_tree_15=prune.tree(full_tree, best = 15)
summary(pruned_tree_15)
plot(pruned_tree_15)
text(pruned_tree_15,pretty=1)

tree_preds <- predict(pruned_tree_15,newdata=data_valid)[,2]

## We only want the Y=1 probability predictions
probs_success_valid=tree_preds


#make binary classifications training 
classifications_success_train <- ifelse(probs_success_train > 0.5, "YES", "NO")
classifications_success_train <- ifelse(is.na(classifications_success_train), "NO", classifications_success_train)
summary(classifications_success_train)

#make binary classifications 
classifications_success_valid <- ifelse(probs_success_valid > .5, "YES", "NO")
classifications_success_valid <- ifelse(is.na(classifications_success_valid), "NO", classifications_success_valid)
summary(classifications_success_valid)


## Training
classifications<-classifications_success_train
classifications_bin<-ifelse(classifications=='YES',1,0)
actuals<-data_train['success']
actuals_bin<-ifelse(actuals$success=='YES',1,0)


## Validation
classifications<-classifications_success_valid
classifications_bin<-ifelse(classifications=='YES',1,0)
actuals<-data_valid['success']
actuals_bin<-ifelse(actuals$success=='YES',1,0)



CM = table(actuals_bin, classifications_bin)


TP <- CM[2,2]
TN <- CM[1,1]
FP <- CM[1,2]
FN <- CM[2,1]

TPR <- TP/(TP+FN)
TNR <- TN/(TN+FP)
FPR <- 1-TNR

accuracy <- (TP+TN)/(TP+TN+FP+FN)



## Attempt 3: Random Forest

rf.mod <- randomForest(success~.,
                       data=data_train,
                       mtry=22,
                       ntree=500,
                       importance=TRUE)

rf_preds <- predict(rf.mod, newdata=data_valid)
rf_acc <- mean(ifelse(rf_preds==data_valid$success,1,0))

rf.mod
rf_acc

## Hyper Parameter tuning 
## Have commented the code because it takes a lot of time to run
## Uncomment before running

#treelength <- c(50,100,200,400,500,600)
#valid_acc <- rep(0, length(treelength))
#train_acc <- rep(0, length(treelength))
#
#for (i in 1:length(treelength)) {
#  rf.mod <- randomForest(success~.,
#                         data=data_train,
#                         mtry=22,
#                         ntree=treelength[i],
#                         importance=TRUE)
#  rf_pred_train <- predict(rf.mod, newdata=data_train)
#  rf_pred_valid <- predict(rf.mod, newdata=data_valid)
#  train_acc[i] <- mean(ifelse(rf_pred_train==data_train$success,1,0))
#  valid_acc[i] <- mean(ifelse(rf_pred_valid==data_valid$success,1,0))
#}
#
#plot(treelength,valid_acc,type="l",col="blue",main="Accuracy-Tree Length",
#     xlab="Tree Length", ylab="Accuracy")
#lines(treelength,train_acc,col="green")
#legend("bottomright",  
#       legend = c("Test", "Train"),
#       col = c("blue", "green"),
#       lwd=2)


## ntree=500 is the best model

rf.mod <- randomForest(success~.,
                       data=data_train,
                       mtry=22,
                       ntree=500,
                       importance=TRUE)

rf_pred_train <- predict(rf.mod, newdata=data_train)
rf_pred_test <- predict(rf.mod, newdata=data_valid)



## Training
classifications<-rf_pred_train
classifications_bin<-ifelse(classifications=='YES',1,0)
actuals<-data_train['success']
actuals_bin<-ifelse(actuals$success=='YES',1,0)


## Validation
classifications<-rf_pred_test
classifications_bin<-ifelse(classifications=='YES',1,0)
actuals<-data_valid['success']
actuals_bin<-ifelse(actuals$success=='YES',1,0)



CM = table(actuals_bin, classifications_bin)


TP <- CM[2,2]
TN <- CM[1,1]
FP <- CM[1,2]
FN <- CM[2,1]

TPR <- TP/(TP+FN)
TNR <- TN/(TN+FP)
FPR <- 1-TNR

accuracy <- (TP+TN)/(TP+TN+FP+FN)


## Attempt 4: XG Boost

dummy <- dummyVars( ~ . , data=data_train)
one_hot_train_data <- data.frame(predict(dummy, newdata =data_train))
train_data_new<-one_hot_train_data%>% select(-c('success.NO','success.YES'))
dtrain = xgb.DMatrix(as.matrix(train_data_new), label=one_hot_train_data$success.YES)


dummy <- dummyVars( ~ . , data=data_valid)
one_hot_valid_data <- data.frame(predict(dummy, newdata =data_valid))
valid_data_new<-one_hot_valid_data%>% select(-c('success.NO','success.YES'))
dvalid = xgb.DMatrix(as.matrix(valid_data_new))


bst <- xgboost(dtrain, max.depth = 2, eta = 0.2, nrounds = 1000,colsample_bylevel=0.25,verbose = 0)




bst_pred <- predict(bst, dvalid)
bst_classifications <- ifelse(bst_pred > 0.5, 'YES', 'NO')
bst_acc <- mean(ifelse(bst_classifications == data_valid$success, 1, 0))
bst_acc

## Feature Importance
library(vip)
vip(bst,num_features = 20)


## Hyper Parameter tuning 
## Have commented the code because it takes a lot of time to run
## Uncomment before running

#grid_search <- function(){
#  
#  #three hyperparameters can possibly really change predictive performance of xgboost (although maybe not)
#  
#  
#  depth_choose <- c(2,3)
#  nrounds_choose <- c(200,500,1000)
#  colsample_bylevel<-c(0.1,0.25,0.4)
#  eta_choose <- c(0.1,0.2,0.3)
#  
#  #nested loops to tune these three parameters
#  print('depth, nrounds, eta,thiscol, accuracy')
#  for(i in c(1:length(depth_choose))){
#    for(j in c(1:length(nrounds_choose))){
#      for(k in c(1:length(eta_choose))){
#        for(l in c(1:length(colsample_bylevel))){
#          thisdepth <- depth_choose[i]
#          thisnrounds <- nrounds_choose[j]
#          thiseta <- eta_choose[k]
#          thiscol<-colsample_bylevel[l]
#          
#          inner_bst <- xgboost(dtrain, max.depth = thisdepth, eta = thiseta, nrounds = thisnrounds, colsample_bylevel=thiscol,verbose = 0)
#          
#          inner_bst_pred <- predict(inner_bst, dvalid)
#          inner_bst_classifications <- ifelse(inner_bst_pred > 0.5, 'YES', 'NO')
#          inner_bst_acc <- mean(ifelse(inner_bst_classifications == data_valid$success, 1, 0))
#          
#          #print the performance for every combination
#          print(paste(thisdepth, thisnrounds, thiseta, inner_bst_acc, sep = ", "))
#        }
#      }
#    }
#  }
#}
#
#grid_search()

## Best Param: max.depth = 2, eta = 0.2, nrounds = 1000,colsample_bylevel=0.25
bst <- xgboost(dtrain, max.depth = 2, eta = 0.2, nrounds = 1000,colsample_bylevel=0.25,verbose = 0)

## Checking Validation confusion matrix and accuracy with the best parameters
bst_pred <- predict(bst, dvalid)

classifications_success_valid <- ifelse(bst_pred > .5, "YES", "NO")
classifications<-classifications_success_valid
classifications_bin<-ifelse(classifications=='YES',1,0)
actuals<-data_valid['success']
actuals_bin<-ifelse(actuals$success=='YES',1,0)



CM = table(actuals_bin, classifications_bin)


TP <- CM[2,2]
TN <- CM[1,1]
FP <- CM[1,2]
FN <- CM[2,1]

TPR <- TP/(TP+FN)
TNR <- TN/(TN+FP)
FPR <- 1-TNR

accuracy <- (TP+TN)/(TP+TN+FP+FN)




## Running the model on testing dataset

test<-test%>%select(-c(success))
dummy <- dummyVars( ~ . , data=test)
one_hot_test_data <- data.frame(predict(dummy, newdata =test))
cols<-intersect(colnames(one_hot_test_data),colnames(train_data_new))
cols2<-colnames(train_data_new%>%select(-cols))
one_hot_test_data<-one_hot_test_data%>%select(colnames(train_data_new))




dtest = xgb.DMatrix(as.matrix(one_hot_test_data))
bst_pred <- predict(bst, dtest)
bst_classifications <- ifelse(bst_pred > 0.5, 'YES', 'NO')
table(bst_classifications)

write.table(bst_classifications, "success_group11.csv", row.names = FALSE)

