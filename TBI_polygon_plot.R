TBI_0 <- read.csv('~/Documents/mindlight/EEG_processing/Simulated_testing/TBI_data/TBI_0.csv', header = F)
TBI_1 <- read.csv('~/Documents/mindlight/EEG_processing/Simulated_testing/TBI_data/TBI_1.csv', header = F)
TBI_2 <- read.csv('~/Documents/mindlight/EEG_processing/Simulated_testing/TBI_data/TBI_2.csv', header = F)
TBI_3 <- read.csv('~/Documents/mindlight/EEG_processing/Simulated_testing/TBI_data/TBI_3.csv', header = F)



# We create 2 vectors x and y. It is a polynomial function.
#x <- runif(300,  min=0, max=1500) 
#y <-  a[1,1]*x^3 + a[1,2]* x^2 + a[1,3]*x + a[1,4]



x=TBI_3[,1]
y=TBI_3[,2]

# Basic plot of x and y :
plot(x,y,col=rgb(0.4,0.2,0.8,0.6), pch=16 , cex=1.3 , xlab="" , ylab="", title="") 

plot_poly(TBI_3, color = .2)
plot_poly(TBI_0, color = .4)
plot_poly(TBI_1, color = .6)
plot_poly(TBI_2, color = .8)




plot_poly <- function(TBI, color=.4){
  x=TBI[,1]
  y=TBI[,2]
  points(x,y, col=rgb(0.4,color,0.8,0.6), pch=16 , cex=1.3)
  # Can we find a polynome that fit this function ?
  model=lm(y ~ x + I(x^2) + I(x^3))
  
  # I can get the features of this model :
  summary(model)
  model$coefficients
  summary(model)$adj.r.squared
  
  #For each value of x, I can get the value of y estimated by the model, and the confidence interval around this value.
  myPredict <- predict( model , interval="predict" )
  
  #Finally, I can add it to the plot using the line and the polygon function with transparency.
  ix <- sort(x,index.return=T)$ix
  lines(x[ix], myPredict[ix , 1], col=2, lwd=2 )  
  polygon(c(rev(x[ix]), x[ix]), c(rev(myPredict[ ix,3]), myPredict[ ix,2]), col = rgb(0.7,color,0.7,0.4) , border = NA)
  
}


