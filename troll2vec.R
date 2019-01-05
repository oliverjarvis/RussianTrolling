#Setuo
setwd("/Users/oliverjarvis/Cognitive Science/Eksamen/")
library(pacman)
p_load(tidyverse, lme4, text2vec, tidytext, Rtsne, gofastr, ngram, ggrepel, vimes,  RFLPtools, proxy, qgraph, ggthemes)

\#Load stop words
data("stop_words")
#Turns stop words into a list
sw <- stop_words$word

#read our pre-prepared tweets
tweets <- read_csv("cleaned_tweets.csv")

#Load content and split into left trolls and right trolls
content <- as.vector(tweets$content)
leftTroll <- subset(tweets, account_category=="LeftTroll")
rightTroll <- subset(tweets, account_category=="RightTroll")

#Set a random seed for reproducibility
set.seed(34)

#Word embeddings for left and for right
#Function to create word embedding
VectorizeTweets <- function(tweets){
  # Create iterator over tokens
  tokens = space_tokenizer(tweets)
  
  # Create vocabulary. Terms will be unigrams (simple words).
  it = itoken(tokens, progressbar = FALSE)
  vocab = create_vocabulary(it, stopwords=sw)
  vocab = prune_vocabulary(vocab, term_count_min = 5L)
  
  # Use our filtered vocabulary
  vectorizer = vocab_vectorizer(vocab)
  
  # use window of 5 for context words
  tcm = create_tcm(it, vectorizer, skip_grams_window = 5L)
  
  glove = GlobalVectors$new(word_vectors_size = 200, vocabulary = vocab, x_max = 10)
  wv_main = glove$fit_transform(tcm, n_iter = 100, convergence_tol = 0.01)
  
  wv_context = glove$components
  
  word_vectors = wv_main + t(wv_context)
  return(word_vectors)
}

#Create a dataset of all left troll tweets
leftTroll <- paste(leftTroll$content, collapse=" ")
leftTroll <- str_remove_all(leftTroll, "\"")

#Create a dataset of right troll tweets
rightTroll <- paste(rightTroll$content, collapse=" ")
rightTroll <- str_remove_all(rightTroll, "\"")

#Create word embedding for left and right troll
leftVec <- VectorizeTweets(leftTroll)
rightVec <- VectorizeTweets(rightTroll)

####CHOOSE OUR CHOSEN WORD
#Change words as necessary
topicWord <- as.character("russia")

#### get lists of top 20 words for left and right for each chosen word

##############################################
######## LEFT TROLL ANALYSIS ################
#############################################

#Get vector
leftT <- leftVec[topicWord, ,drop = FALSE]
#Get cosine similarities
cos_sim = sim2(x = leftVec, y = leftT, method = "cosine", norm = "l2")

#choose top 20 words
left_terms <- head(sort(cos_sim[,1], decreasing = TRUE), 20)

#create new dataframe from words, and tell the origin
left_terms_df <- data.frame(value = left_terms, VectorOrigin = "LeftTroll")

#Add word column
left_terms_df$word <- rownames(left_terms_df)

#Select relevant rows
left_terms_df <- select(left_terms_df, VectorOrigin, word)

#Set index
rownames(left_terms_df) <- c(1:20)


##############################################
######## RIGHT TROLL ANALYSIS ################
##############################################

#Get vector
rightT <- rightVec[topicWord, ,drop = FALSE]
#Get cosine similarities
cos_sim = sim2(x = rightVec, y = rightT, method = "cosine", norm = "l2")

#choose top 20 words
right_terms <- head(sort(cos_sim[,1], decreasing = TRUE), 20)

#create new dataframe from words, and tell the origin
right_terms_df <- data.frame(value = right_terms, VectorOrigin = "rightTroll")

#Add word column
right_terms_df$word <- rownames(right_terms_df)

#Select relevant rows
right_terms_df <- select(right_terms_df, VectorOrigin, word)

#Set index
rownames(right_terms_df) <- c(1:20)

##############################################
######## BOTH TROLL ANALYSIS ################
##############################################

both_df <- bind_rows(left_terms_df, right_terms_df)

#Measure word distances to leftVec
both_df$left_dist <- NA
for(i in 1:nrow(both_df)){
  if(as.character(both_df[i,2]) %in% rownames(leftVec)){
    term1 = leftVec[topicWord, ,drop = FALSE]
    term2 = leftVec[as.character(both_df[i,2]), ,drop = FALSE]
    both_df[i,3] <- sim2(x = term1, y = term2, method = "cosine", norm = "l2")
  }
}

#Measure word distances to rightVec
both_df$right_dist <- NA
for(i in 1:nrow(both_df)){
  if(as.character(both_df[i,2]) %in% rownames(rightVec)){
    term1 = rightVec[topicWord, ,drop = FALSE]
    term2 = rightVec[as.character(both_df[i,2]), ,drop = FALSE]
    both_df[i,4] <- sim2(x = term1, y = term2, method = "cosine", norm = "l2")
  }
}

#Find duplicates
both_df <- group_by(both_df, word) %>% mutate(unique = n() == 1)

#Add both dataframes to each vector
vector_gun <- both_df
vector_gun <- vector_gun[-21,]
vector_terrorism <- both_df
vector_terrorism <- vector_terrorism[-21,]
vector_jobs <- both_df
vector_jobs <- vector_jobs[-21,]
vector_obamacare <- both_df
vector_obamacare <- vector_obamacare[-21,]
vector_russia <- both_df
vector_russia <- vector_russia[-21,]
vector_gun <- ungroup(vector_gun)
vector_terrorism <- ungroup(vector_terrorism)
vector_jobs <- ungroup(vector_jobs)
vector_obamacare <- ungroup(vector_obamacare)
vector_russia <- ungroup(vector_russia)

#Save all our computed instances
#Future todo: automize with loop
write_csv(vector_gun, "vector_gun.csv")
write_csv(vector_terrorism, "vector_terrorism.csv")
write_csv(vector_jobs, "vector_jobs.csv")
write_csv(vector_russia, "vector_russia.csv")
write_csv(vector_obamacare, "vector_obamacare.csv")


#Put each left/right into their own dataframe, with only the origin vector and distances selected
vector_gun <- as_data_frame(vector_gun)
vector_gun.left <-arrange(vector_gun, VectorOrigin, desc(left_dist))
vector_gun.right <-arrange(vector_gun, VectorOrigin, desc(right_dist))

vector_terrorism <- as_data_frame(vector_terrorism)
vector_terrorism.left<-arrange(vector_terrorism, VectorOrigin, desc(left_dist))
vector_terrorism.right<-arrange(vector_terrorism, VectorOrigin, desc(right_dist))

vector_russia <- as_data_frame(vector_russia)
vector_russia.left<-arrange(vector_russia, VectorOrigin, desc(left_dist))
vector_russia.right<-arrange(vector_russia, VectorOrigin, desc(right_dist))

vector_jobs <- as_data_frame(vector_jobs)
vector_jobs.left<-arrange(vector_jobs, VectorOrigin, desc(left_dist))
vector_jobs.right<-arrange(vector_jobs, VectorOrigin, desc(right_dist))

vector_obamacare <- as_data_frame(vector_obamacare)
vector_obamacare.left<-arrange(vector_obamacare, VectorOrigin, desc(left_dist))
vector_obamacare.right<-arrange(vector_obamacare, VectorOrigin, desc(right_dist))

#Plot semantic relationship for left
left_matrix.2 <- normalize_input(leftVec)
left_tsne = Rtsne(left_matrix.2, check_duplicates=FALSE, pca=T, perplexity=10, theta=0.2, dims=2)
left_tsne = as.data.frame(left_tsne$Y)
left_tsne$word <- rownames(leftVec)
left_tsne_points <- subset(left_tsne, word=="gun" | word=="russia" | word=="terrorism" | word == "jobs" | word=="obamacare")

p = ggplot(left_tsne, aes(V1, V2), label=word) +
  geom_point(size = 0.1) + 
  geom_point(data = left_tsne_points, color="blue", size=5) +
  geom_label(data=left_tsne_points, aes(x = V1, y = V2, label=c("gun", "russia", "terrorism", "jobs", "obamacare")), colour = "black", alpha=1, size=4, fontface="bold", inherit.aes = FALSE) +
  theme(legend.position = "bottom") + 
  labs(title="Left Troll Vector")
p
ggsave("left-troll-vector.png", p)

#geom_label_repel(aes(label = word), color = 'white', size = 3.5)

#Plot semantic relationship for right
right_matrix.2 <- normalize_input(rightVec)
right_tsne = Rtsne(right_matrix.2, check_duplicates=FALSE, pca=T, perplexity=10, theta=0.2, dims=2)
right_tsne = as.data.frame(right_tsne$Y)
right_tsne$word <- rownames(rightVec)

right_tsne_points <- subset(right_tsne, word=="gun" | word=="russia" | word=="terrorism" | word == "jobs" | word=="obamacare")

p = ggplot(right_tsne, aes(V1, V2), label=word) +
  geom_point(size = 0.1) + 
  geom_point(data = right_tsne_points, color="blue", size=5) +
  geom_label(data=right_tsne_points, aes(x = V1, y = V2, label=c("gun", "russia", "terrorism", "jobs", "obamacare")), colour = "black", alpha=1, size=4, fontface="bold", inherit.aes = FALSE) +
  theme(legend.position = "bottom") + 
  labs(title="Right Troll Vector")
p
ggsave("right-troll-vector-label.png", p)


vector_gun.left.viz <- vector_gun %>% filter(!is.na(vector_gun$left_dist))
vector_gun.left.viz <- select(vector_gun.left.viz, VectorOrigin, word, left_dist)
vector_gun.left.viz$word <- paste(vector_gun.left.viz$word, as.character(round(as.numeric(vector_gun.left.viz$left_dist), 2)))
colnames(vector_gun.left.viz) <- c("group", "individual", "value")
vector_gun.left.viz$value <- ((-(vector_gun.left.viz$value))+1)*100
data=vector_gun.left.viz
data$group[data$group == "rightTroll"] <- "ArightTroll"
data = data[-1,]
data$group = as.factor(data$group)

#Set a number of 'empty bar' to add at the end of each group
empty_bar=3
#Add these bars
to_add = data.frame( matrix(NA, empty_bar*nlevels(data$group), ncol(data)) )
#Set the colnames of to_add to the colnames of data ???
colnames(to_add) = colnames(data)
#Set the groups to the groups from data
to_add$group=rep(levels(data$group), each=empty_bar)
#Bind data and rbind
data=rbind(data, to_add)
#arrange by group
data=data %>% arrange(group)
#give each row an id of its row number
data$id=seq(1, nrow(data))

# Get the name and the y position of each label
label_data=data

#variable set
number_of_bar=nrow(label_data)

#Give each bar an angle based on some math
angle= 90 - 360 * (label_data$id-0.5) /number_of_bar     # I substract 0.5 because the letter must have the angle of the center of the bars. Not extreme right(1) or extreme left (0)

#Calculate more stuff
label_data$hjust<-ifelse( angle < -90, 1, 0)
label_data$angle<-ifelse(angle < -90, angle+180, angle)

# prepare a data frame for base lines
base_data=data %>% 
  group_by(group) %>% 
  summarize(start=min(id), end=max(id) - empty_bar) %>% 
  rowwise() %>% 
  mutate(title=mean(c(start, end)))

# prepare a data frame for grid (scales)
grid_data = base_data
grid_data$end = grid_data$end[ c( nrow(grid_data), 1:nrow(grid_data)-1)] + 1
grid_data$start = grid_data$start - 1
grid_data=grid_data[-1,]

# Make the plot
p = ggplot(data, aes(x=as.factor(id), y=value, fill=group)) +       # Note that id is a factor. If x is numeric, there is some space between the first bar
  
  geom_bar(aes(x=as.factor(id), y=value, fill=group), stat="identity", alpha=0.5) +
  
  # Add a val=100/75/50/25 lines. I do it at the beginning to make sur barplots are OVER it.
  geom_segment(data=grid_data, aes(x = end, y = 200, xend = start, yend = 200), colour = "grey", alpha=1, size=0.3 , inherit.aes = FALSE ) +
  geom_segment(data=grid_data, aes(x = end, y = 150, xend = start, yend = 150), colour = "grey", alpha=1, size=0.3 , inherit.aes = FALSE ) +
  geom_segment(data=grid_data, aes(x = end, y = 100, xend = start, yend = 100), colour = "grey", alpha=1, size=0.3 , inherit.aes = FALSE ) +
  geom_segment(data=grid_data, aes(x = end, y = 50, xend = start, yend = 50), colour = "grey", alpha=1, size=0.3 , inherit.aes = FALSE ) +
  geom_segment(data=grid_data, aes(x = end, y = 0, xend = start, yend = 0), colour = "grey", alpha=1, size=0.3 , inherit.aes = FALSE ) +
  
  # Add text showing the value of each 100/75/50/25 lines
  annotate("text", x = rep(max(data$id),3), y = c(0, 100, 200), label = c("1", "0", "-1") , color="grey", size=3 , angle=0, fontface="bold", hjust=1) +
  
  geom_bar(aes(x=as.factor(id), y=value, fill=group), stat="identity", alpha=0.5) +
  ylim(-100,300) +
  theme_minimal() +
  theme(
    legend.position = "none",
    axis.text = element_blank(),
    axis.title = element_blank(),
    panel.grid = element_blank(),
    plot.margin = unit(rep(-1,4), "cm") 
  ) +
  scale_fill_manual(values=c("#c72127", "#61acf0")) +
  coord_polar() + 
  geom_text(data=label_data, aes(x=id, y=value+10, label=individual, hjust=hjust), color="black", fontface="bold",alpha=0.6, size=2.5, angle= label_data$angle, inherit.aes = FALSE ) +
  
  # Add base line information
  geom_segment(data=base_data, aes(x = start, y = -5, xend = end, yend = -5), colour = "black", alpha=0.8, size=0.6 , inherit.aes = FALSE )  +
  geom_text(data=base_data, aes(x = title, y = -18, label=c("Right words", "Left Words")), hjust=c(1,0), colour = "black", alpha=0.8, size=2.5, fontface="bold", inherit.aes = FALSE) +
  annotate("text", x = 0, y = -100, label = "gun")

p
ggsave("gun-leftvec.png", p)


plotAndWork <- function(vector_we, lbl){
  
  vector_we.viz <- vector_we %>% filter(!is.na(vector_we$left_dist))
  vector_we.viz <- select(vector_we.viz, VectorOrigin, word, left_dist)
  vector_we.viz$word <- paste(vector_we.viz$word, as.character(round(as.numeric(vector_we.viz$left_dist), 2)))
  colnames(vector_we.viz) <- c("group", "individual", "value")
  vector_we.viz$value <- ((-(vector_we.viz$value))+1)*100
  data=vector_we.viz
  data$group[data$group == "rightTroll"] <- "ArightTroll"
  data = data[-1,]
  data$group = as.factor(data$group)
  # Create dataset
  #data=data.frame(
  #  #Create each value
  #  individual=paste( "Mister ", seq(1,20), sep=""),
  #  #Create two groups of 10 values
  #  group=c( rep('Left Vector', 10), rep('Right Vector', 10)) ,
  #  #Give each row a random value between 10 and a 100
  #  value=sample( seq(10,), 20, replace=T)
  #)
  
  
  
  #Set a number of 'empty bar' to add at the end of each group
  empty_bar=3
  #Add these bars
  to_add = data.frame( matrix(NA, empty_bar*nlevels(data$group), ncol(data)) )
  #Set the colnames of to_add to the colnames of data ???
  colnames(to_add) = colnames(data)
  #Set the groups to the groups from data
  to_add$group=rep(levels(data$group), each=empty_bar)
  #Bind data and rbind
  data=rbind(data, to_add)
  #arrange by group
  data=data %>% arrange(group, value)
  #give each row an id of its row number
  data$id=seq(1, nrow(data))
  
  # Get the name and the y position of each label
  label_data=data
  
  #variable set
  number_of_bar=nrow(label_data)
  
  #Give each bar an angle based on some math
  angle= 90 - 360 * (label_data$id-0.5) /number_of_bar     # I substract 0.5 because the letter must have the angle of the center of the bars. Not extreme right(1) or extreme left (0)
  
  #Calculate more stuff
  label_data$hjust<-ifelse( angle < -90, 1, 0)
  label_data$angle<-ifelse(angle < -90, angle+180, angle)
  
  # prepare a data frame for base lines
  base_data=data %>% 
    group_by(group) %>% 
    summarize(start=min(id), end=max(id) - empty_bar) %>% 
    rowwise() %>% 
    mutate(title=mean(c(start, end)))
  
  # prepare a data frame for grid (scales)
  grid_data = base_data
  grid_data$end = grid_data$end[ c( nrow(grid_data), 1:nrow(grid_data)-1)] + 1
  grid_data$start = grid_data$start - 1
  grid_data=grid_data[-1,]
  
  # Make the plot
  p = ggplot(data, aes(x=as.factor(id), y=value, fill=group)) +       # Note that id is a factor. If x is numeric, there is some space between the first bar
    labs(title = "Left Vector") + 
    geom_bar(aes(x=as.factor(id), y=value, fill=group), stat="identity", alpha=0.5) +
    
    # Add a val=100/75/50/25 lines. I do it at the beginning to make sur barplots are OVER it.
    geom_segment(data=grid_data, aes(x = end, y = 200, xend = start, yend = 200), colour = "grey", alpha=1, size=0.3 , inherit.aes = FALSE ) +
    geom_segment(data=grid_data, aes(x = end, y = 150, xend = start, yend = 150), colour = "grey", alpha=1, size=0.3 , inherit.aes = FALSE ) +
    geom_segment(data=grid_data, aes(x = end, y = 100, xend = start, yend = 100), colour = "grey", alpha=1, size=0.3 , inherit.aes = FALSE ) +
    geom_segment(data=grid_data, aes(x = end, y = 50, xend = start, yend = 50), colour = "grey", alpha=1, size=0.3 , inherit.aes = FALSE ) +
    geom_segment(data=grid_data, aes(x = end, y = 0, xend = start, yend = 0), colour = "grey", alpha=1, size=0.3 , inherit.aes = FALSE ) +
    
    # Add text showing the value of each 100/75/50/25 lines
    annotate("text", x = rep(max(data$id),3), y = c(0, 100, 200), label = c("1", "0", "-1") , color="grey", size=3 , angle=0, fontface="bold", hjust=1) +
    
    geom_bar(aes(x=as.factor(id), y=value, fill=group), stat="identity", alpha=0.5) +
    ylim(-100,300) +
    theme_minimal() +
    theme(
      legend.position = "none",
      axis.text = element_blank(),
      axis.title = element_blank(),
      panel.grid = element_blank(),
      plot.margin = unit(rep(-1,4), "cm") 
    ) +
    scale_fill_manual(values=c("#c72127", "#61acf0")) +
    coord_polar() + 
    geom_text(data=label_data, aes(x=id, y=value+10, label=individual, hjust=hjust), color="black", fontface="bold",alpha=0.6, size=2, angle= label_data$angle, inherit.aes = FALSE ) +
    
    # Add base line information
    geom_segment(data=base_data, aes(x = start, y = -5, xend = end, yend = -5), colour = "black", alpha=0.8, size=0.6 , inherit.aes = FALSE )  +
    geom_text(data=base_data[1,], aes(x = nrow(data)/2/2, y = -38, label=c("Right words")), colour = "black", alpha=1.0, size=2, fontface="bold", inherit.aes = FALSE, angle=-90) +
    geom_text(data=base_data[2,], aes(x = nrow(data)/2/2*3, y = -38, label=c("Left words")), colour = "black", alpha=1.0, size=2, fontface="bold", inherit.aes = FALSE, angle=90) +
    geom_text(x = 0, y = -100, label = lbl, size=2)
}

\

p <- plotAndWork(vector_russia, "russia")
ggsave("russia-leftvec.png", p)

p <- plotAndWork(vector_jobs, "jobs")
ggsave("jobs-leftvec.png", p)

p <- plotAndWork(vector_gun, "gun")
ggsave("gun-leftvec.png", p)

p <- plotAndWork(vector_obamacare, "obamacare")
ggsave("obamacare-leftvec.png", p)

p <- plotAndWork(vector_terrorism, "terrorism")
ggsave("terrorism-leftvec.png", p)

###RIGHT
p <- plotAndWork(vector_russia, "russia")
ggsave("russia-rightvec.png", p)

p <- plotAndWork(vector_jobs, "jobs")
ggsave("jobs-rightvec.png", p)

p <- plotAndWork(vector_gun, "gun")
ggsave("gun-rightvec.png", p)

p <- plotAndWork(vector_obamacare, "obamacare")
ggsave("obamacare-rightvec.png", p)

p <- plotAndWork(vector_terrorism, "terrorism")
ggsave("terrorism-rightvec.png", p)

