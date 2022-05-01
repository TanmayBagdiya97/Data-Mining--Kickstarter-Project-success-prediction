#load libraries
library(tidyverse)
library(lubridate)
library(text2vec)
library(tm)
library(tidyr)
library(readr)
library(tree)


#load data files
train_x <- read_csv("ks_training_X.csv")
train_y <- read_csv("ks_training_y.csv")
test_x <- read_csv("ks_test_X.csv")


#join the training y to the training x file
#also turn two of the target variables into factors

train <- train_x %>%
  left_join(train_y, by = "id") %>%
  group_by(category_name) %>%
  mutate(count=n())%>%
  ungroup()%>%
  mutate(category_name=if_else(count<20,'Other',category_name))%>%
  mutate(location_slug =if_else(location_slug=="None",region,str_sub(location_slug, start= -2)))%>%
  mutate(smiling_project=if_else(smiling_project>100,100,round(smiling_project,0)),
        smiling_creator=if_else(smiling_creator>100,100,round(smiling_creator,0)))%>%
  mutate(isbwImg1=ifelse (is.na(isbwImg1),'NULL',isbwImg1),
         color_foreground=ifelse (is.na(color_foreground),'NULL',color_foreground),
         color_background=ifelse (is.na(color_background),'NULL',color_background),
         isTextPic=ifelse (is.na(isTextPic),'NULL',isTextPic),
         isLogoPic=ifelse (is.na(isLogoPic),'NULL',isLogoPic),
         isCalendarPic=ifelse (is.na(isCalendarPic),'NULL',isCalendarPic),
         isDiagramPic=ifelse (is.na(isDiagramPic),'NULL',isDiagramPic),
         isShapePic=ifelse (is.na(isShapePic),'NULL',isShapePic)
  )%>%
  mutate(afinn_pos_norm=round(afinn_pos/num_words,1),
             afinn_neg_norm=round(afinn_neg/num_words,1))%>%
  mutate(success = as.factor(success),
         big_hit = as.factor(big_hit),
         creator_id= as.factor(creator_id),
         numfaces_project = as.factor(numfaces_project),
         numfaces_creator = as.factor(numfaces_creator),
         male_project = as.factor(male_project),
         male_creator = as.factor(male_creator),
         female_project=as.factor(female_project),
         female_creator = as.factor(female_creator),
         isTextPic = as.factor(isTextPic),
         isLogoPic = as.factor(isLogoPic),
         isCalendarPic = as.factor(isCalendarPic),
         isDiagramPic = as.factor(isDiagramPic),
         isShapePic=as.factor(isShapePic)) 

a<-stringr::str_extract_all(train$reward_amounts, "\\d+")

remove_zero<- function(a){
  a[seq_along(a)%% 2==1]
}

a<-lapply(a,remove_zero)

a<-lapply(a,as.numeric)
train$avg_reward_amt<-unlist(lapply(a, mean))
  
train['days_active']<-difftime(train$deadline,train$launched_at,units='days')

train['launch_month']<-month(train$launched_at)



# Iterate over the individual documents and convert them to tokens
# Uses the functions defined above.
prep_fun = tolower
cleaning_tokenizer <- function(v) {
  v %>%
    removeNumbers %>% #remove all numbers
    removePunctuation %>%
    removeWords(stopwords(kind="en")) %>% #remove stopwords
    stemDocument %>%
    word_tokenizer 
}
tok_fun = cleaning_tokenizer

# Iterate over the individual documents and convert them to tokens
# Uses the functions defined above.
it_train = itoken(train$blurb, 
                  preprocessor = prep_fun, 
                  tokenizer = tok_fun, 
                  ids = train$id, 
                  progressbar = FALSE)

# Create the vocabulary from the itoken object
vocab = create_vocabulary(it_train)
vocab_small = prune_vocabulary(vocab, vocab_term_max = 500)

# Create a vectorizer object using the vocabulary we learned
vectorizer = vocab_vectorizer(vocab_small)

# Convert the training documents into a DTM and make it a binary BOW matrix
dtm_train = create_dtm(it_train, vectorizer)
dim(dtm_train)
dtm_train_bin <- dtm_train>0+0


# Convert the small sparse matrix into a dense one
dense = as.matrix(dtm_train_bin)+0

dense = subset(dense, select=-c(get("goal")))

dense = subset(dense, select=-c(get("success")))
# Use cbind() to append the columns
train <- cbind(train, dense)

train_success <- train %>%
  select(-c(id,big_hit, backers_count,creator_name,captions,name,blurb,tag_names,created_at,accent_color,category_parent,isbwImg1,accent_color,color_foreground,color_background,avg_wordlengths,sentence_counter,avgsentencelength,avgsyls,reward_descriptions,deadline,launched_at,count,reward_amounts,afinn_pos,afinn_neg))


train_inst = sample(nrow(train_success), .7*nrow(train_success))

data_train <- train_success[train_inst,]
data_valid <- train_success[-train_inst,]
write.table(data_train, "data_train.csv", row.names = FALSE)

## Attempt 1:
# EXAMPLE PREDICTIONS FOR Y = SUCCESS

#create a simple model to predict success and generate predictions in the test data



logistic_success <- glm(success~goal, data = data_train, family = "binomial")
probs_success_train <- predict(logistic_success, newdata = data_train%>%select(-success), type = "response")
probs_success_valid <- predict(logistic_success, newdata = data_valid%>%select(-success), type = "response")

probs_success_test <- predict(logistic_success, newdata = test_x, type = "response")

#make binary classifications training (make sure to check for NAs!)
classifications_success_train <- ifelse(probs_success_train > 0.45, "YES", "NO")
classifications_success_train <- ifelse(is.na(classifications_success_train), "NO", classifications_success_train)
summary(classifications_success_train)

#make binary classifications (make sure to check for NAs!)
classifications_success_valid <- ifelse(probs_success_valid > .45, "YES", "NO")
classifications_success_valid <- ifelse(is.na(classifications_success_valid), "NO", classifications_success_valid)
summary(classifications_success_valid)


accuracy <- function(classifications, actuals){
  correct_classifications <- ifelse(classifications == actuals, 1, 0)
  acc <- sum(correct_classifications)/length(classifications)
  return(acc)}

## Training
classifications<-classifications_success_train
classifications_bin<-ifelse(classifications=='YES',1,0)
actuals<-data_train['success']
actuals_bin<-ifelse(actuals$success=='YES',1,0)
accuracy(classifications,actuals)

## Validation
classifications<-classifications_success_valid
classifications_bin<-ifelse(classifications=='YES',1,0)
actuals<-data_valid['success']
actuals_bin<-ifelse(actuals$success=='YES',1,0)
accuracy(classifications,actuals)


CM = table(actuals_bin, classifications_bin)


TP <- CM[2,2]
TN <- CM[1,1]
FP <- CM[1,2]
FN <- CM[2,1]

TPR <- TP/(TP+FN)
TNR <- TN/(TN+FP)
FPR <- 1-TNR

accuracy <- (TP+TN)/(TP+TN+FP+FN)

table(actuals)



## Attempt 2: Logistic Regression
# EXAMPLE PREDICTIONS FOR Y = SUCCESS

#create a simple model to predict success and generate predictions in the test data



logistic_success <- glm(success~goal, data = data_train, family = "binomial")
probs_success_train <- predict(logistic_success, newdata = data_train%>%select(-success), type = "response")
probs_success_valid <- predict(logistic_success, newdata = data_valid%>%select(-success), type = "response")

probs_success_test <- predict(logistic_success, newdata = test_x, type = "response")

#make binary classifications training (make sure to check for NAs!)
classifications_success_train <- ifelse(probs_success_train > 0.45, "YES", "NO")
classifications_success_train <- ifelse(is.na(classifications_success_train), "NO", classifications_success_train)
summary(classifications_success_train)

#make binary classifications (make sure to check for NAs!)
classifications_success_valid <- ifelse(probs_success_valid > .45, "YES", "NO")
classifications_success_valid <- ifelse(is.na(classifications_success_valid), "NO", classifications_success_valid)
summary(classifications_success_valid)


accuracy <- function(classifications, actuals){
  correct_classifications <- ifelse(classifications == actuals, 1, 0)
  acc <- sum(correct_classifications)/length(classifications)
  return(acc)}

## Training
classifications<-classifications_success_train
classifications_bin<-ifelse(classifications=='YES',1,0)
actuals<-data_train['success']
actuals_bin<-ifelse(actuals$success=='YES',1,0)
accuracy(classifications,actuals)

## Validation
classifications<-classifications_success_valid
classifications_bin<-ifelse(classifications=='YES',1,0)
actuals<-data_valid['success']
actuals_bin<-ifelse(actuals$success=='YES',1,0)
accuracy(classifications,actuals)


CM = table(actuals_bin, classifications_bin)


TP <- CM[2,2]
TN <- CM[1,1]
FP <- CM[1,2]
FN <- CM[2,1]

TPR <- TP/(TP+FN)
TNR <- TN/(TN+FP)
FPR <- 1-TNR

accuracy <- (TP+TN)/(TP+TN+FP+FN)

table(actuals)



## Attempt 3: Decision Trees
# EXAMPLE PREDICTIONS FOR Y = SUCCESS

#create a simple model to predict success and generate predictions in the test data
library(rpart)
library(caret)

accuracy <- function(classifications, actuals){
  correct_classifications <- ifelse(classifications == actuals, 1, 0)
  acc <- sum(correct_classifications)/length(classifications)
  return(acc)
}

mycontrol = tree.control(nobs = nrow(train_success), mincut = 1, minsize = 2, mindev = 0.00005)
full_tree <- tree(success ~ . ,
                  data = train_success, 
                  control = mycontrol)



treelength <- c(2, 4, 6, 8, 10, 15, 20, 25, 30, 35,40)
test_acc <- rep(0, length(treelength))
train_acc <- rep(0, length(treelength))

for (i in 1:length(treelength)) {
  pruned_tree=prune.tree(full_tree, best =treelength[i])
  tree_pred_train <- predict(pruned_tree,newdata=data_train)[,2]
  tree_pred_test <- predict(pruned_tree,newdata=data_valid)[,2]
  tree_pred_train <- ifelse(tree_pred_train>0.5,'YES','NO')
  tree_pred_test <- ifelse(tree_pred_test>0.5,'YES','NO')
  train_acc[i] <- accuracy(tree_pred_train,data_train$success)
  test_acc[i] <- accuracy(tree_pred_test,data_valid$success)
}

plot(treelength,train_acc,type="l",col="blue",main="Accuracy-Tree Length",
     xlab="Tree Length", ylab="Accuracy")
lines(treelength,test_acc,col="green")
legend("bottomright",  
       legend = c("Train", "Test"),
       col = c("blue", "green"),
       lwd=2)



pruned_tree_15=prune.tree(full_tree, best = 10)
summary(pruned_tree_15)
plot(pruned_tree_15)
text(pruned_tree_15,pretty=1)

tree_preds <- predict(pruned_tree_15,newdata=data_valid)[,2]

#look at tree_preds - what is this representation
tree_preds

## We only want the Y=1 probability predictions
probs_success_valid=tree_preds




#make binary classifications training (make sure to check for NAs!)
classifications_success_train <- ifelse(probs_success_train > 0.5, "YES", "NO")
classifications_success_train <- ifelse(is.na(classifications_success_train), "NO", classifications_success_train)
summary(classifications_success_train)

#make binary classifications (make sure to check for NAs!)
classifications_success_valid <- ifelse(probs_success_valid > .5, "YES", "NO")
classifications_success_valid <- ifelse(is.na(classifications_success_valid), "NO", classifications_success_valid)
summary(classifications_success_valid)


accuracy <- function(classifications, actuals){
  correct_classifications <- ifelse(classifications == actuals, 1, 0)
  acc <- sum(correct_classifications)/length(classifications)
  return(acc)}

## Training
classifications<-classifications_success_train
classifications_bin<-ifelse(classifications=='YES',1,0)
actuals<-data_train['success']
actuals_bin<-ifelse(actuals$success=='YES',1,0)
accuracy(classifications,actuals)

## Validation
classifications<-classifications_success_valid
classifications_bin<-ifelse(classifications=='YES',1,0)
actuals<-data_valid['success']
actuals_bin<-ifelse(actuals$success=='YES',1,0)
accuracy(classifications,actuals)


CM = table(actuals_bin, classifications_bin)


TP <- CM[2,2]
TN <- CM[1,1]
FP <- CM[1,2]
FN <- CM[2,1]

TPR <- TP/(TP+FN)
TNR <- TN/(TN+FP)
FPR <- 1-TNR

accuracy <- (TP+TN)/(TP+TN+FP+FN)

table(actuals)


probs_success_test <- predict(pruned_tree_15,newdata=test_x)[,2]
classifications_success<-ifelse(probs_success_test>0.5,'YES','NO')
write.table(classifications_success, "success_group11.csv", row.names = FALSE)


## Attempt 4: Random Forest
# EXAMPLE PREDICTIONS FOR Y = SUCCESS

#create a simple model to predict success and generate predictions in the test data
library(randomForest)
library(caret)



rf.mod <- randomForest(success~goal+category_parent+location_type,
                       data=data_train,
                       mtry=2, ntree=200,
                       importance=TRUE)

rf_preds <- predict(rf.mod, newdata=data_valid)
rf_acc <- mean(ifelse(rf_preds==data_valid$success,1,0))

rf.mod
rf_acc



accuracy <- function(classifications, actuals){
  correct_classifications <- ifelse(classifications == actuals, 1, 0)
  acc <- sum(correct_classifications)/length(classifications)
  return(acc)
}

mycontrol = tree.control(nobs = nrow(data_train), mincut = 1, minsize = 2, mindev = 0.00005)
full_tree <- tree(success ~ goal+as.factor(category_parent)+as.factor(location_type),
                  data = data_train, 
                  control = mycontrol)



treelength <- c(2, 4, 6, 8, 10, 15, 20, 25, 30, 35,40)
test_acc <- rep(0, length(treelength))
train_acc <- rep(0, length(treelength))

for (i in 1:length(treelength)) {
  pruned_tree=prune.tree(full_tree, best =treelength[i])
  tree_pred_train <- predict(pruned_tree,newdata=data_train)[,2]
  tree_pred_test <- predict(pruned_tree,newdata=data_valid)[,2]
  tree_pred_train <- ifelse(tree_pred_train>0.5,'YES','NO')
  tree_pred_test <- ifelse(tree_pred_test>0.5,'YES','NO')
  train_acc[i] <- accuracy(tree_pred_train,data_train$success)
  test_acc[i] <- accuracy(tree_pred_test,data_valid$success)
}

plot(treelength,train_acc,type="l",col="blue",main="Accuracy-Tree Length",
     xlab="Tree Length", ylab="Accuracy")
lines(treelength,test_acc,col="green")
legend("bottomright",  
       legend = c("Train", "Test"),
       col = c("blue", "green"),
       lwd=2)



pruned_tree_15=prune.tree(full_tree, best = 10)
summary(pruned_tree_15)
plot(pruned_tree_15)
text(pruned_tree_15,pretty=1)

tree_preds <- predict(pruned_tree_15,newdata=data_valid)[,2]

#look at tree_preds - what is this representation
tree_preds

## We only want the Y=1 probability predictions
probs_success_valid=tree_preds


#make binary classifications training (make sure to check for NAs!)
classifications_success_train <- ifelse(probs_success_train > 0.5, "YES", "NO")
classifications_success_train <- ifelse(is.na(classifications_success_train), "NO", classifications_success_train)
summary(classifications_success_train)

#make binary classifications (make sure to check for NAs!)
classifications_success_valid <- ifelse(probs_success_valid > .5, "YES", "NO")
classifications_success_valid <- ifelse(is.na(classifications_success_valid), "NO", classifications_success_valid)
summary(classifications_success_valid)


accuracy <- function(classifications, actuals){
  correct_classifications <- ifelse(classifications == actuals, 1, 0)
  acc <- sum(correct_classifications)/length(classifications)
  return(acc)}

## Training
classifications<-classifications_success_train
classifications_bin<-ifelse(classifications=='YES',1,0)
actuals<-data_train['success']
actuals_bin<-ifelse(actuals$success=='YES',1,0)
accuracy(classifications,actuals)

## Validation
classifications<-classifications_success_valid
classifications_bin<-ifelse(classifications=='YES',1,0)
actuals<-data_valid['success']
actuals_bin<-ifelse(actuals$success=='YES',1,0)
accuracy(classifications,actuals)


CM = table(actuals_bin, classifications_bin)


TP <- CM[2,2]
TN <- CM[1,1]
FP <- CM[1,2]
FN <- CM[2,1]

TPR <- TP/(TP+FN)
TNR <- TN/(TN+FP)
FPR <- 1-TNR

accuracy <- (TP+TN)/(TP+TN+FP+FN)

table(actuals)



# EXAMPLE PREDICTIONS FOR Y = BACKERS_COUNT

#create a simple model to predict backers_count and generate predictions in the test data
train_backers <- train %>%
  select(-c(big_hit, success))

linear_backers <- lm(backers_count~goal, data = train_backers)
preds_backers <- predict(linear_backers, newdata = test_x)


# EXAMPLE PREDICTIONS FOR Y = BIG_HIT

#create a simple model to predict success and generate predictions in the test data
train_bighit <- train %>%
  select(-c(success, backers_count))

logistic_bighit <- glm(big_hit~goal, data = train_bighit, family = "binomial")
probs_bighit <- predict(logistic_bighit, newdata = test_x, type = "response")



#output your predictions
#they must be in EXACTLY this format
#a .csv file with the naming convention targetvariable_groupAAA.csv, where you replace targetvariable with your chosen target, and AAA with your group name
#in exactly the same order as they are in the test_x file

# For success, each row should be a binary YES (is successful) or NO (not successful)
# For backers count, each row should be a number representing the estimated number of backers
# For big_hit, each row should be a number between 0 and 1 representing the estimated probability of big_hit

#this code creates sample outputs in the correct format
write.table(classifications_success, "success_group0.csv", row.names = FALSE)
write.table(preds_backers, "backers_count_group0.csv", row.names = FALSE)
write.table(probs_bighit, "big_hit_group0.csv", row.names = FALSE)

# I have evaluated these predictions against the test set
# the success predictions have accuracy = 0.6045278
# the backers_count predictions have RMSE = 396.0263
# the big_hit predictions have AUC = 0.6303743

# You should be able to improve upon these without much effort!