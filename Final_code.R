#load libraries
library(tidyverse)
library(lubridate)
library(text2vec)
library(tm)
library(tidyr)
library(readr)
library(tree)
library(caret)
library(xgboost)
library(vip)

#load data files
train_x <- read_csv("ks_training_X.csv")
train_y <- read_csv("ks_training_y.csv")
test_x <- read_csv("ks_test_X.csv")
train_x <-rbind(train_x,test_x)



#join the training y to the training x file
#also turn two of the target variables into factors

normalize <- function(x) {
  return ((x - mean(x)) / sd(x))
}


## Creating average reward amount
a<-stringr::str_extract_all(train_x$reward_amounts, "\\d+")
remove_zero<- function(a){
  a[seq_along(a)%% 2==1]
}
a<-lapply(a,remove_zero)
a<-lapply(a,as.numeric)
train_x$avg_reward_amt<-unlist(lapply(a, mean))


train_x['days_active']<-difftime(train_x$deadline,train_x$launched_at,units='days')
train_x['launch_month']<-month(train_x$launched_at)




train <- train_x %>%
  left_join(train_y, by = "id") %>%
  group_by(category_name) %>%
  mutate(category_name=ifelse(n()<500,'Other',category_name)) %>%
  ungroup() %>%
  group_by(category_parent) %>%
  mutate(category_parent=ifelse(n()<500,'Other',category_parent)) %>%
  ungroup() %>%
  group_by(region) %>%
  mutate(region=ifelse(n()<500,'Other',region)) %>%
  ungroup() %>%
  group_by(location_type) %>%
  mutate(location_type=ifelse(n()<500,'Other',location_type)) %>%
  ungroup() %>%
  mutate(location_slug =ifelse(location_slug=="None","US",str_sub(location_slug, start= -2)))%>%
  group_by(location_slug) %>%
  mutate(location_slug=ifelse(n()<500,'Other',location_slug)) %>%
  ungroup() %>%
  mutate(smiling_project=ifelse(smiling_project>100,100,round(smiling_project,0)),
         smiling_creator=ifelse(smiling_creator>100,100,round(smiling_creator,0)))%>%
  mutate(isbwImg1=ifelse (is.na(isbwImg1),0,isbwImg1),
         color_foreground=ifelse (is.na(color_foreground),'White',color_foreground),
         color_background=ifelse (is.na(color_background),'Black',color_background),
         isTextPic=ifelse (is.na(isTextPic),0,isTextPic),
         isLogoPic=ifelse (is.na(isLogoPic),0,isLogoPic),
         isCalendarPic=ifelse (is.na(isCalendarPic),0,isCalendarPic),
         isDiagramPic=ifelse (is.na(isDiagramPic),0,isDiagramPic),
         isShapePic=ifelse (is.na(isShapePic),0,isShapePic)
         
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
         ADJ=ifelse (is.na(round(ADJ/num_words,4)),0,round(ADJ/num_words,4)),
         goal=normalize(goal),
         numfaces_project=normalize(numfaces_project),
         numfaces_creator=normalize(numfaces_creator),
         male_project=normalize(male_project),
         male_creator=normalize(male_creator),
         female_project=normalize(female_project),
         female_creator=normalize(female_creator),
         smiling_project=normalize(smiling_project),
         smiling_creator=normalize(smiling_creator),
         minage_project=normalize(minage_project),
         minage_creator=normalize(minage_creator),
         maxage_project=normalize(maxage_project),
         maxage_creator=normalize(maxage_creator),
         num_words=normalize(num_words),
         grade_level=normalize(grade_level),
         avg_reward_amt=normalize(avg_reward_amt),
         days_active=normalize(days_active),
         perday=normalize(perday),
         avg_wordlengths=normalize(avg_wordlengths),
         sentence_counter=normalize(sentence_counter),
         avgsentencelength=normalize(avgsentencelength),
         avgsyls=normalize(avgsyls),
         
         
         
         goal=ifelse(goal>3,3,goal),
         numfaces_project=ifelse(numfaces_project>3,3,numfaces_project),
         numfaces_creator=ifelse(numfaces_creator>3,3,numfaces_creator),
         male_project=ifelse(male_project>3,3,male_project),
         male_creator=ifelse(male_creator>3,3,male_creator),
         female_project=ifelse(female_project>3,3,female_project),
         female_creator=ifelse(female_creator>3,3,female_creator),
         smiling_project=ifelse(smiling_project>3,3,smiling_project),
         smiling_creator=ifelse(smiling_creator>3,3,smiling_creator),
         minage_project=ifelse(minage_project>3,3,minage_project),
         minage_creator=ifelse(minage_creator>3,3,minage_creator),
         maxage_project=ifelse(maxage_project>3,3,maxage_project),
         maxage_creator=ifelse(maxage_creator>3,3,maxage_creator),
         num_words=ifelse(num_words>3,3,num_words),
         grade_level=ifelse(grade_level>3,3,grade_level),
         avg_reward_amt=ifelse(avg_reward_amt>3,3,avg_reward_amt),
         days_active=ifelse(days_active>3,3,days_active),
         perday=ifelse(perday>3,3,perday),
         avg_wordlengths=ifelse(avg_wordlengths>3,3,avg_wordlengths),
         sentence_counter=ifelse(sentence_counter>3,3,sentence_counter),
         avgsentencelength=ifelse(avgsentencelength>3,3,avgsentencelength),
         avgsyls=ifelse(avgsyls>3,3,avgsyls)
         )


#created_at
train_success <- train %>%
  select(-c(id,big_hit,creator_id, backers_count,creator_name,captions,name,blurb,tag_names,accent_color,accent_color,reward_descriptions,deadline,launched_at,reward_amounts,afinn_pos,afinn_neg,color_foreground))

dummy <- dummyVars( ~ .+goal:category_parent , data=train_success)
train_success <- data.frame(predict(dummy, newdata =train_success))
test_x <-train_success[97421:108728,]
train_success <- train_success[0:97420,]

train_inst = sample(nrow(train_success), .8*nrow(train_success))

data_train <- train_success[train_inst,]
data_valid <- train_success[-train_inst,]


## XGB

train_data_new<-data_train%>% select(-c('success.NO','success.YES'))
dtrain = xgb.DMatrix(as.matrix(train_data_new), label=data_train$success.YES)


valid_data_new<-data_valid%>% select(-c('success.NO','success.YES'))
dvalid = xgb.DMatrix(as.matrix(valid_data_new))


bst <- xgboost(dtrain, max.depth = 3, eta = 0.2, nrounds = 1000,colsample_bylevel=0.2, verbose = 0)

bst_pred <- predict(bst, dvalid)
bst_classifications <- ifelse(bst_pred > 0.5, 1, 0)
bst_acc <- mean(ifelse(bst_classifications == data_valid$success.YES, 1, 0))
bst_acc

vip(bst,num_features = 20)

#grid_search <- function(){
#  
#  depth_choose <- c(3)
#  nrounds_choose <- c(1000)
#  colsample_bylevel<-c(0.1,0.2,0.3)
#  eta_choose <- c(0.1,0.2,0.3)
#  #nested loops to tune these three parameters
#  print('depth, nrounds, thiscol,eta, accuracy')
#  for(i in c(1:length(depth_choose))){
#    for(j in c(1:length(nrounds_choose))){
#      for(k in c(1:length(eta_choose))){
#        for(l in c(1:length(colsample_bylevel))){
#        thisdepth <- depth_choose[i]
#        thisnrounds <- nrounds_choose[j]
#        thiseta <- eta_choose[k]
#        thiscol<-colsample_bylevel[l]
#
#        
#        inner_bst <- xgboost(dtrain, max.depth = thisdepth, eta = thiseta, nrounds = thisnrounds, colsample_bylevel=thiscol,verbose = 0)
#        
#        inner_bst_pred <- predict(inner_bst, dvalid)
#        inner_bst_classifications <- ifelse(inner_bst_pred > 0.5, 1, 0)
#        inner_bst_acc <- mean(ifelse(inner_bst_classifications == data_valid$success.YES, 1, 0))
#        
#        #print the performance for every combination
#        print(paste(thisdepth, thisnrounds,thiscol, thiseta, inner_bst_acc, sep = ", "))
#        }
#      }
#    }
#  }
#}

#grid_search()


## Training on all data
train_final<-train_success%>% select(-c('success.NO','success.YES'))
dtrain_final = xgb.DMatrix(as.matrix(train_final), label=train_success$success.YES)

bst <- xgboost(dtrain_final, max.depth = 3, eta = 0.2, nrounds = 1000,colsample_bylevel=0.2, verbose = 0)



test_x<-test_x%>%select(-c(success.YES,success.NO))
dtest = xgb.DMatrix(as.matrix(test_x))
bst_pred <- predict(bst, dtest)
bst_classifications <- ifelse(bst_pred > 0.5, 'YES', 'NO')
table(bst_classifications)

write.table(bst_classifications, "success_group11.csv", row.names = FALSE)

