# loading libraries 
library(factoextra)
library(class) # for KNN
library(caret) # for confusion matrix 

# Reading in iris data
iris.raw = read.csv('iris.data', header=FALSE)

# Define min-max normalization function
min_max_norm <- function(x) {
  (x - min(x)) / (max(x) - min(x))
  }
# Normalize first four columns in iris data set
iris.norm = as.data.frame(lapply(iris.raw[1:4], min_max_norm))

# creating scatter plot matrix of normalized iris data
pairs(iris.norm, main='Scatter Plot Matrix of Normalized Iris Data')

###################################
##### Hierarchical Clustering #####
###################################

# Computing distance matrix using euclidean distance 
d = dist(iris.norm, method="euclidean",diag=T,upper=T)

# Single linkage method (Nearest Neighbour)
hc_single <- hclust(d=d,method="single")
cutree(hc_single, k=3)
fviz_dend(hc_single, cex=0.5, k=3, main="Nearest Neighbour",
          color_labels_by_k=TRUE, rect=TRUE)

# Complete linkage method (Farthest Neighbour)
hc_complete <- hclust(d=d,method="complete")
cutree(hc_complete, k=3) 
fviz_dend(hc_complete, cex=0.5, k=3, main="Farthest Neighbour",
          color_labels_by_k=TRUE, rect=TRUE)

#  Average linkage method (Group Average)
hc_avg <- hclust(d=d,method="average")
cutree(hc_avg, k=3) 
fviz_dend(hc_avg, cex=0.5, k=3, main="Group Average",
          color_labels_by_k=TRUE, rect=TRUE)

# Ward's method
hc_ward <- hclust(d=d,method="ward.D")
cutree(hc_ward, k=3) 
fviz_dend(hc_ward, cex=0.5, k=3, main="Ward's",
          color_labels_by_k=TRUE, rect=TRUE)

# scatter plot matrix using Farthest Neighbour 
pairs(iris.norm, pch=21, bg=c("red","blue", "green")[cutree(hc_complete, k=3)],
      oma=c(4,4,4,10))
legend("topright", legend=c("Cluster 1","Cluster 2", "Cluster 3"),
       fill=c("red","blue", "green"),xpd=NA,cex=0.4)


##############################
##### K-Means Clustering #####
##############################

# elbow method
fviz_nbclust(iris.norm, kmeans, method = "wss")

# Average silhouette for kmeans
fviz_nbclust(iris.norm, kmeans, method = "silhouette")

# K-means into 3 clusters
km_3 <- kmeans(iris.norm, centers=3) # centers = 3 -> 3 clusters
km_3$cluster # which observations belong to which cluster
km_3$size # number of observations in each cluster

# scatter plot matrix using k-means clustering
pairs(iris.norm, pch=21, bg=c("red","blue","green")[km_3$cluster],
      oma=c(4,4,4,10))
legend("topright", legend=c("Cluster 1","Cluster 2", "Cluster 3"),
       fill=c("red","blue","green"),xpd=NA,cex=0.5)

# Visualize k-means clusters by performing PCA and plot according to first 2 PCs
pc <- prcomp(iris.norm, center=T,scale=T)
eigen <- (pc$sdev)^2 # eigenvalues/variances
pvar= eigen/sum(eigen) # individual contribution for each variable i.e. variance
csum <- cumsum(pvar) # cumulative contribution for each variable
table <- cbind(eigen, pvar, csum)
table
# the average of the eigenvalues 
sum(eigen)/4
# scree plot to help determine number of components
screeplot(pc,type="lines", main="Screeplot")
# eigenvectors for the principal components we retain
pc$rotation[,1:2]

# Coordinates of individuals
ind.coord <- as.data.frame(pc$x)
# Add clusters obtained using the K-means 
ind.coord$cluster <- factor(km_3$cluster)
# plot PC1 vs PC2
plot(x=ind.coord$PC1, y=ind.coord$PC2, cex=2,pch=19,
     col=c("red","Yellow","green")[ind.coord$cluster],
     xlab="PC1 (72.77%)", ylab="PC2 (23.03%)")
legend("topright", legend=c("Cluster 1","Cluster 2", "Cluster 3"),
       fill=c("red","Yellow","green"),xpd=NA,cex=0.5)
abline(v=0,h=0)
# labeling points with their cluster assignment 
text(PC2~PC1, labels=rownames(ind.coord), data=ind.coord,cex=0.9,font=0.5)

fviz_cluster(km_3, data = iris.norm)

################################
##### K-Nearest Neighbours #####
################################

# random selection of 70% data.
iris.ind <- sample(1:nrow(iris.norm), size=nrow(iris.norm)*0.7,replace = FALSE) 
# split data into training and test sets
train.iris <- iris.norm[iris.ind,] # 70% training data
test.iris <- iris.norm[-iris.ind,] # 30% test data

# Creating seperate dataframe for verification, i.e. our target 
train.iris_labels <- iris.raw[iris.ind,5]
test.iris_labels <- iris.raw[-iris.ind,5]

# Take square root of the number of observations to find optimal k value 
sqrt(nrow(iris.norm)) 
# implementing knn model 
knn.12 <- knn(train=train.iris, test=test.iris, cl=train.iris_labels, k=12)

# Checking accuracy of our KNN model
confusionMatrix(table(knn.12 ,test.iris_labels))

# Calculating accuracy of k values from 1 to 13. 
acc.k <- 1
k <- 1
for (i in 1:13){
  knn.model <- knn(train=train.iris, test=test.iris, cl=train.iris_labels, k=i)
  acc.k[i] <- 100 * sum(test.iris_labels == knn.model)/NROW(test.iris_labels)
  k=i
  cat(k,'=',acc.k[i],'
')
}

#Accuracy plot
plot(acc.k, type="b", xlab="k",ylab="Accuracy")
