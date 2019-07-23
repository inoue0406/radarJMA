# check grid position

library("ggplot2")
theme_set(theme_bw())
library("sf")

library("rnaturalearth")
library("rnaturalearthdata")

setwd("C:/Users/Tsuyoshi Inoue/GoogleDrive/Research/radarJMA/plotting")

world <- ne_countries(scale = "medium", returnclass = "sf")
class(world)

df <- read.csv("alljapan_grid_info.csv",encoding = "")
# only 100%
df <- df[df$data_ratio == 1.0,]

# create polygon data to match ggplot2 format
values <- data.frame(
  id = c(1:nrow(df)),
  value = df$data_ratio
)
positions <- data.frame(
  id = rep(values$id, each = 4),
  x = array(rbind(df$lon0,df$lon0,df$lon1,df$lon1)),
  y = array(rbind(df$lat0,df$lat1,df$lat1,df$lat0))
)
datapoly <- merge(values, positions, by = c("id"))

ggplot(data = world) +
  geom_sf() +
  coord_sf(xlim = c(123, 148), ylim = c(22, 47), expand = FALSE) + 
  geom_polygon(data=datapoly, linetype=1, color="black", aes(x=x,y=y,group=id,fill=value), alpha=0.1)
