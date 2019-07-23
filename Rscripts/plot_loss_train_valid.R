
setwd("C:/Users/Tsuyoshi Inoue/GoogleDrive/Research/radarJMA/plotting")

case = "result_20190717_clstm_tryoverfit"

fpath <- sprintf("result/%s/train_batch.log",case)
print(paste("reading",fpath))
df <- read.table(fpath,header = TRUE,sep="\t")

#
date = "190718_alljapan"

# 
dir.create(sprintf("png/%s",date),showWarnings = FALSE)

# loop through thresholds
vars <- c("RMSE","CSI","POD","FAR","Cor","MaxMSE","FSS_mean")
vlims <- c(100,1,1,1,1,2000,1)
for(k in 1:length(vars)){
  var <- vars[k]
  for(th in c(0.5,10,20)){
    png(sprintf("png/%s/%s_compare_%d%s_%.1f.png",date,date,k,var,th), 
        width = 500, height = 500)
    title <- sprintf("%s : threshold=%.2f",var,th)
    plot(1,type='n',
         xlab="prediction time [min]",
         ylab=var,ylim=c(0,vlims[k]),xlim=c(0,60),main=title,
         cex.lab=1.5, cex.axis=1.5, cex.main=1.5, cex.sub=1.5)
    grid()
    legend("topright", legend = df_s$lbl, col = df_s$color, lty = df_s$lty, lwd = df_s$lwd)
    for(n in 1:nrow(df_s)){
      fname <- sprintf("test_evaluation_predtime_%.2f.csv",th)
      fpath <- sprintf("result/%s/%s",df_s$case[n],fname)
      print(paste("reading",fpath))
      df <- read.csv(fpath)
      # plot a line
      lines(df$tpred_min,df[[var]],type="l",col=df_s$color[n],lwd=df_s$lwd[n],
            lty=as.integer(df_s$lty[n]))
    }
    dev.off()
  }
}
