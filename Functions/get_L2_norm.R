GetL2Norm <- function(vec){
  sqrt(t(vec)%*%vec)
}