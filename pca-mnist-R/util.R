eucledian_distance <- function(x1, x2){ 
    data = (x1 - x2) ^ 2
    return(sqrt(sum(data)))
}

knn_based_classifier <- function(data, training_image, training_label){
    colnames = c("distance", "label")
    distance_matrix = matrix(nrow=nrow(training_image), ncol=2, dimnames=list(NULL, colnames))

    for(i in 1:nrow(training_image)){
        if( ! identical(data, training_image[i,]) ){

            distance <- eucledian_distance(data,training_image[i,])
            label <- training_label[i]
            distance_matrix[i,] = c(distance, label)
        } 
    }
    
    # // TODO SORT matrix
    distance_matrix = distance_matrix[ order(distance_matrix[, 1]) , ]

    # // GET 15 NEAREST NEIGHBOUR
    nearest_neighbour = distance_matrix[1:15,]
    # print(nearest_neighbour)

    # cat("REAL LABEL : ", training_label[1], "\n")

    # // CLASSIFY IT
    # label_matrix = matrix(nrow=10, ncol=2, dimnames=list(NULL, colnames))
    # label = which.max(table(nearest_neighbour))
    label = names(which.max(table(nearest_neighbour[,2])))
    # cat("PREDICTION LABEL : ", label, "\n")


    return(label)

}
