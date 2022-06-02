#install.packages(c("tm", SnowballC", "wordcloud", "proxy", "kernlab", "NLP", "openNLP","ggplot2"))
#install.packages("CORElearn")
#install.packages("ipred")
library(ggplot2)
# We also have to install models for the English language
#install.packages("openNLPmodels.en", repos="http://datacube.wu.ac.at/", type="source")
#install.packages("stringr")                       # Install stringr package
library("stringr")


train_data = read.table(
  file = "train_data.tsv",
  sep = "\t",
  header = TRUE,
  comment.char = "",
  quote = ""
)
fake_news_train = train_data$label
train_data = train_data$text_a

test_data = read.table(
  file = "test_data.tsv",
  sep = "\t",
  header = TRUE,
  comment.char = "",
  quote = ""
)
fake_news_test = test_data$label
test_data = test_data$text_a




#(  _ \(  _ \( ___)___(  _ \(  _ \(  _  )/ __)( ___)/ __)/ __)(_  _)( \( )/ __)
# )___/ )   / )__)(___))___/ )   / )(_)(( (__  )__) \__ \\__ \ _)(_  )  (( (_-.
#(__)  (_)\_)(____)   (__)  (_)\_)(_____)\___)(____)(___/(___/(____)(_)\_)\___/

library(tm)

# Construct a corpus for a vector as input.
corpus_test <- Corpus(VectorSource(test_data))
corpus_train <- Corpus(VectorSource(train_data))


subSpace <-
  content_transformer(function(x, pattern)
    gsub(pattern, " ", x))

# Remove links
corpus_test = tm_map(corpus_test, subSpace, "https[^ ]*")
corpus_train = tm_map(corpus_train, subSpace, "https[^ ]*")

# Remove everything except letters
corpus_test <- tm_map(corpus_test, subSpace, "[^a-zA-Z]")
corpus_train <- tm_map(corpus_train, subSpace, "[^a-zA-Z]")

# Change letters to lower case
corpus_test <- tm_map(corpus_test, content_transformer(tolower))
corpus_train <- tm_map(corpus_train, content_transformer(tolower))

# Remove stopwords 
corpus_test <-
  tm_map(corpus_test, removeWords, stopwords('english'))
corpus_train <-
  tm_map(corpus_train, removeWords, stopwords('english'))

# Read the custom stopwords list
conn = file("english.stop.txt", open = "r")
mystopwords = readLines(conn)
close(conn)

# Remove stopwords
corpus_test <- tm_map(corpus_test, removeWords, mystopwords)
corpus_train <- tm_map(corpus_train, removeWords, mystopwords)

# Stem words to retrieve their radicals
corpus_test <- tm_map(corpus_test, stemDocument)
corpus_train <- tm_map(corpus_train, stemDocument)

# Strip extra whitespace from text documents
corpus_test <- tm_map(corpus_test, stripWhitespace)
corpus_train <- tm_map(corpus_train, stripWhitespace)




tdm_train <- TermDocumentMatrix(corpus_train)
tdm_test <- TermDocumentMatrix(corpus_test)


termFrequency_train <- rowSums(as.matrix(tdm_train))
termFrequency_train <-
  subset(termFrequency_train, termFrequency_train >= 300)
qplot(seq(length(termFrequency_train)),
      sort(termFrequency_train),
      xlab = "index",
      ylab = "Freq")

termFrequency_test <- rowSums(as.matrix(tdm_test))
termFrequency_test <-
  subset(termFrequency_test, termFrequency_test >= 300)
qplot(seq(length(termFrequency_test)),
      sort(termFrequency_test),
      xlab = "index",
      ylab = "Freq")


library(wordcloud)

mat_train <- as.matrix(tdm_train)
wordFreq_train <- sort(rowSums(mat_train), decreasing = TRUE)
grayLevels_train <-
  gray((wordFreq_train + 10) / (max(wordFreq_train) + 10))
wordcloud(
  words = names(wordFreq_train),
  freq = wordFreq_train,
  min.freq = 100,
  random.order = F,
  colors = grayLevels_train
)

mat_test <- as.matrix(tdm_test)
wordFreq_test <- sort(rowSums(mat_test), decreasing = TRUE)
grayLevels_test <-
  gray((wordFreq_test + 10) / (max(wordFreq_test) + 10))
wordcloud(
  words = names(wordFreq_test),
  freq = wordFreq_test,
  min.freq = 100,
  random.order = F,
  colors = grayLevels_test
)




#__________           _____                                                  _____                    __________
#___  ____/__________ __  /____  _____________     ____________________________  /___________  _________  /___(_)____________
#__  /_   _  _ \  __ `/  __/  / / /_  ___/  _ \    _  ___/  __ \_  __ \_  ___/  __/_  ___/  / / /  ___/  __/_  /_  __ \_  __ \
#_  __/   /  __/ /_/ // /_ / /_/ /_  /   /  __/    / /__ / /_/ /  / / /(__  )/ /_ _  /   / /_/ // /__ / /_ _  / / /_/ /  / / /
#/_/      \___/\__,_/ \__/ \__,_/ /_/    \___/     \___/ \____//_/ /_//____/ \__/ /_/    \__,_/ \___/ \__/ /_/  \____//_/ /_/


dtm_train <-
  DocumentTermMatrix(corpus_train, control = list(weighting = weightTfIdf))
dtm_train <- removeSparseTerms(dtm_train, sparse = 0.99)
train_mat <- as.matrix(dtm_train)
train_df <- as.data.frame(train_mat)


dtm_test <-
  DocumentTermMatrix(corpus_test,
                     control = list(dictionary = Terms(dtm_train), weighting = weightTfIdf))
test_mat <- as.matrix(dtm_test)
train_names <- names(train_df)
test_mat <- test_mat[, train_names]
test_df <- as.data.frame(test_mat)



train_df <- cbind(train_df, fake_news_train)
names(train_df)[ncol(train_df)] <- "fake_news_train"

test_df <- cbind(test_df, fake_news_test)
names(test_df)[ncol(test_df)] <- "fake_news_test"



#   _____             .___     .__  .__
#  /     \   ____   __| _/____ |  | |__| ____    ____
# /  \ /  \ /  _ \ / __ |/ __ \|  | |  |/    \  / ___\
#/    Y    (  <_> ) /_/ \  ___/|  |_|  |   |  \/ /_/  >
#\____|__  /\____/\____ |\___  >____/__|___|  /\___  /
#       \/             \/    \/             \//_____/


# RANDOM FOREST

library(CORElearn)

cm.rf <- CoreModel(fake_news_train ~ ., data = train_df, model = "rf")
rf.predicted <- predict(cm.rf, test_df, type = "class")
rf.observed <- test_df$fake_news_test
rf <- table(rf.observed, rf.predicted)

rf_ca <- sum(diag(rf)) / sum(rf)
rf_recall <- rf[1, 1] / sum(rf[1, ])
rf_precision <- rf[1, 1] / sum(rf[, 1])

# NAIVE BAYES

library(CORElearn)

cm.nb <-
  CoreModel(fake_news_train ~ ., data = train_df, model = "bayes")
nb.predicted <- predict(cm.nb, test_df, type = "class")
nb.observed <- test_df$fake_news_test
nb <- table(nb.observed, nb.predicted)

nb_ca <- sum(diag(rb)) / sum(nb)
nb_recall <- nb[1, 1] / sum(nb[1, ])
nb_precision <- nb[1, 1] / sum(nb[, 1])


# KNN

library(class)

train_r <- which(names(train_df) == "fake_news_train")
test_r <- which(names(test_df) == "fake_news_test")

predicted <-
  knn(train_df[,-train_r], test_df[, -test_r], train_df$fake_news_train, k=5)
observed <- test_df$fake_news_test
knn <- table(observed, predicted)

knn_ca <- sum(diag(knn)) / sum(knn)
knn_recall <- knn[1, 1] / sum(knn[1, ])
knn_precision <- knn[1, 1] / sum(knn[, 1])


# SUPPORT-VECTOR MACHINE

library(caret)
library(e1071)


set.seed(112233)
library(parallel)
# Calculate the number of cores
no_cores <- detectCores() - 1

library(doParallel)
# create the cluster for caret to use
cl <- makePSOCKcluster(no_cores)
registerDoParallel(cl)


# svm with a linear kernel
as.factor(fake_news_train)
model.svm <- train(as.factor(fake_news_train) ~ .,
                   data = train_df,
                   method = "svmLinear")

svm.predicted <- predict(model.svm, test_df, type = "raw")
svm.observed <- test_df$fake_news_test
svm <- table(svm.observed, svm.predicted)

svm_ca <- sum(diag(svm)) / sum(svm)
svm_recall <- svm[1, 1] / sum(svm[1, ])
svm_precision <- svm[1, 1] / sum(svm[, 1])


# BAGGING

library(ipred)

bagging <- bagging(fake_news_train ~ ., train_df, nbagg = 20)
bag.predicted <- predict(bagging, test_df, type = "class")$class
bag.observed <- test_df$fake_news_test
bag <- table(bag.observed, bag.predicted)

bag_ca <- sum(diag(bag)) / sum(bag)
bag_recall <- bag[1, 1] / sum(bag[1, ])
bag_precision <- bag[1, 1] / sum(bag[, 1])


# MAJORITY

maj.train <- table(train_df$fake_news_train)

maj.class <- 0
if (maj.train[1] < maj.train[2]) {
  maj.class <- 1
}

maj.observed <- test_df$fake_news_test
maj.predicted <- rep(maj.class, length(maj.observed))
maj <- table(maj.observed, maj.predicted)

maj_ca <- maj[2, 1] / sum(maj)
maj_recall <- maj[1, 1] / maj[1, ]
maj_precision <- maj[2, 1] / sum(maj[, 1])


#    ___ __ __   ____  _      __ __   ____  ______  ____  ___   ____
#   /  _]  |  | /    || |    |  |  | /    ||      ||    |/   \ |    \
#  /  [_|  |  ||  o  || |    |  |  ||  o  ||      | |  ||     ||  _  |
# |    _]  |  ||     || |___ |  |  ||     ||_|  |_| |  ||  O  ||  |  |
# |   [_|  :  ||  _  ||     ||  :  ||  _  |  |  |   |  ||     ||  |  |
# |     |\   / |  |  ||     ||     ||  |  |  |  |   |  ||     ||  |  |
# |_____| \_/  |__|__||_____| \__,_||__|__|  |__|  |____|\___/ |__|__|




f1 <- function(recall, precision) {
  2 * ((precision * recall) / (precision + recall))
}

maj_f1 <- f1(maj_recall, maj_precision)
rf_f1 <- f1(rf_recall, rf_precision)
rb_f1 <- f1(rb_recall, rb_precision)
knn_f1 <- f1(knn_recall, knn_precision)
svm_f1 <- f1(svm_recall, svm_precision)
bag_f1 <- f1(bag_recall, bag_precision)



performances <- c(maj_ca, rf_ca, rb_ca, knn_ca, svm_ca, bag_ca)
f1 <- c(maj_f1, rf_f1, rb_f1, knn_f1, svm_f1, bag_f1)
algo.names <-
  c("Majority", "RF", "Naive Bayes", "KNN", "SVM", "Bagging")
ensemble.model <- as.factor(c(0, 1, 0, 0, 0, 1))
result.vec <- data.frame(performances, f1, algo.names, ensemble.model)
reordering <- order(result.vec$performances)
result.vec <- result.vec[reordering, ]
rownames(result.vec) <- NULL

library(ggplot2)
positions <- as.vector(result.vec$algo.names)
ggplot(data = result.vec,
       aes(x = algo.names, y = performances, color = ensemble.model)) +
  geom_point(size = 3, shape = 0) +
  scale_x_discrete(limits = positions) +
  ylim(0.5, 1) +
  xlab("Ensemble type") +
  ylab("Accuracy") +
  geom_hline(yintercept = max(performances), color = "darkgreen") +
  geom_hline(yintercept = min(performances), color = "black") +
  title("Performance comparison") +
  geom_text(label = positions,
            nudge_x = 0,
            nudge_y = -0.01) +
  theme_bw() +
  theme(
    axis.title.x = element_blank(),
    axis.text.x = element_blank(),
    axis.ticks.x = element_blank()
  )

