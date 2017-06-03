eucledian_distance <- function(x1, x2){ 
    data = (x1 - x2) ^ 2
    return(sqrt(sum(data)))
}

prototype_based_classifier <- function(data, models){
    best_distance <- 99999999
    best_label <- NA
    for(i in 1:10){
        distance <- eucledian_distance(data, models[i,])
        if(distance < best_distance){
            best_distance <- distance
            best_label <- i
        }
    }
    
    return(best_label - 1)
}

knn_based_classifier <- function(data, training_image, training_label){
    best_distance <- 99999999
    best_label <- NA
    for(i in 1:2500){
        distance <- eucledian_distance(data,training_image[i,])
        if(distance < best_distance){
            best_distance <- distance
            best_label <- i
        }
    }

    return(training_label[best_label])

}

get_accuracy <- function(confusion_matrix){
    top <- 0
    for(i in 1:10){
        top <- top + confusion_matrix[i,i]
        TP = confusion_matrix[i,i]
        prcession =  TP / sum(confusion_matrix[i,])
        recall = TP / sum(confusion_matrix[,i])
        f1_score = (2 * prcession * recall) / (prcession + recall)

        cat("Precission Class", i-1, "=", prcession, "\n")
        cat("Recall     Class", i-1, "=", recall, "\n")
        cat("F1 Score   Class", i-1, "=", f1_score, "\n")

    }
    overall_accuracy = top / sum(confusion_matrix)
    cat("Overal Accuracy =", overall_accuracy, "\n")
    cat("Error Rate = ", 1 - overall_accuracy, "\n")

}