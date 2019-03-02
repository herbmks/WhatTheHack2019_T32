
getwd()
setwd("/Users/BHM/Documents/Degree_MSc/Year_1/Semester_2/WhatTheHack/WhatTheHack2019_T32/nl")
getwd()



temp = list.files(pattern="*.csv")
for (i in 1:length(temp)) assign(temp[i], read.csv(temp[i]))






library(dplyr)
library(readr)
df <- list.files(full.names = TRUE) %>% 
  lapply(read_csv) %>% 
  bind_cols

colnames(df) <- c(1:838)


pca <- prcomp(df, scale = TRUE)

sapply(df, function(x){is.na(x)})



