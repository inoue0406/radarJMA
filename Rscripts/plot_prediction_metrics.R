
setwd("C:/Users/Tsuyoshi Inoue/GoogleDrive/Research/radarJMA/plotting")

df_s <- data.frame(case=character(),#fname=character(),
                         color=integer(),lwd=integer(),lty=integer(),
                         lbl=character(),
                         stringsAsFactors=FALSE)

# list of caseses to draw                 col lwd lty
df_s[1,] <- c("result_20190527_pers_fss_th05",1,1,2,"persistence")
df_s[2,] <- c("result_20190530_rainym_th05_fss",1,1,1,"rainymotion")
df_s[3,] <- c("result_20190625_clstm_lrdecay07_ep20",2,1,1,"convLSTM-lrdecay")
#df_s[4,] <- c("result_20190704_mclstm_local_rerun_ep20",3,1,1,"convLSTM-2lyr2")
#df_s[5,] <- c("result_20190708_clstm_weightedmse",4,1,1,"convLSTM-wloss")
df_s[4,] <- c("result_20190709_clstm_flatsampled",2,1,2,"convLSTM-flatsampled")
df_s[5,] <- c("result_20190710_clstm_alljapan",3,1,2,"convLSTM-alljapan")
df_s[6,] <- c("result_20190712_tr_clstm_flatsampled",3,1,1,"convLSTM-transfer")
df_s[7,] <- c("result_20190715_clstm_alljapan_2yr",4,1,2,"convLSTM-alljapan2")
df_s[8,] <- c("result_20190716_clstm_tr_2yr_flat",4,1,1,"convLSTM-transfer2")
# 
df_s$color <- as.integer(df_s$color)
df_s$lwd <- as.integer(df_s$lwd)
df_s$lty <- as.integer(df_s$lty)

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
