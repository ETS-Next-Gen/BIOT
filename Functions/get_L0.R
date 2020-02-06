GetL0 <- function(W){
  # W: regression weights
  
  sum(sign(abs(W)))
}