library('rugarch')

setwd("~/Desktop/NetEase2020/data/breakouts_influence_w30min300")

myGarch <- function(filename){
  # read data
  data0.df <- read.csv(filename,header = TRUE)
  data.df = subset(data0.df,attr!='breakout')
  
  # dummy variable
  dummy <- data.df$attr=='after'
  dummy.mtx <- as.matrix(dummy)
  
  
  # set garch model specification
  ## mean.model: mu + dummy
  ## variance.model: garch(1,1) + dummy
  
  dummy2.garch11.spec = ugarchspec(
    variance.model=list(model='sGARCH',garchOrder=c(1,1),external.regressors=dummy.mtx),
    mean.model=list(armaOrder=c(1,0),external.regressors=dummy.mtx)
  )
  
  # fit garch model
  dummy2.garch11.fit = ugarchfit(
    spec=dummy2.garch11.spec, 
    data=data.df$value
  )
  

  # output
  #pvalue <- pt(dummy2.garch11.fit@fit[["tval"]],length(data.df$value)-1,lower.tail = FALSE)
  #res <- cbind(dummy2.garch11.fit@fit[["coef"]],1-pvalue)
  filenum <- strsplit(filename,'\\.')[[1]][1]
  #cat(res, file=paste('./R_params/',filenum,'.txt',sep=''))
  sink(paste('./R_results/',filenum,'.txt',sep=''))
  show(dummy2.garch11.fit)
  sink()
}

# 读取数据并输出结果
path = '.'
files = dir(path)
for (k in 1:length(files)){
  if (grep('csv',files[k])){
    myGarch(files[k]) # 核心函数
  }
}
