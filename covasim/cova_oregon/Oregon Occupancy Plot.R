library(ggplot2)
library(plyr)
library(dplyr)
library(scales)
library(readxl)
library(reshape2)
library(gridExtra)

setwd("C:/Users/Brittany/Google Drive/COVID-19/Modeling Results")
SETLABEL <- "Oregon3152020"
INPUTSET <- "314_OREGON_CONSTRAINED_TRIALS"

#Load CSVs
INPT_OCCUP <- read.csv("MODELING RESULTSSS_R_INPT_OCCUPANCY_END_OF_WK_314_OREGON_UNCONSTRAINED_TRIALS.csv")
ICU_OCCUP <- read.csv("MODELING RESULTSSS_R_ICU_OCCUPANCY_END_OF_WK_314_OREGON_UNCONSTRAINED_TRIALS.csv")
MEDSURG_OCCUP <- read.csv(paste("MODELING RESULTSSS_R_MEDSURG_OCCUPANCY_END_OF_WK_314_OREGON_UNCONSTRAINED_TRIALS.csv"))

scenario_list <- c("Status Quo","Aggressive","Schools Stay Closed","Schools Reopen")
value_list <- c("LB","EV","UB")

INPT_OCCUP$Scenario <- ""
INPT_OCCUP$Type <- ""
ICU_OCCUP$Scenario <- ""
ICU_OCCUP$Type <- ""
MEDSURG_OCCUP$Scenario <- ""
MEDSURG_OCCUP$Type <- ""


rownum = 1
for (i in 1:length(scenario_list)){
  for (j in 1:length(value_list)){
    for (k in 1:10){
      INPT_OCCUP$Scenario[rownum] = scenario_list[i]
      INPT_OCCUP$Type[rownum] = value_list[j]
      ICU_OCCUP$Scenario[rownum] = scenario_list[i]
      ICU_OCCUP$Type[rownum] = value_list[j]
      MEDSURG_OCCUP$Scenario[rownum] = scenario_list[i]
      MEDSURG_OCCUP$Type[rownum] = value_list[j]
      rownum=rownum+1
    }
  }
}

INPT_OCCUP <- subset(INPT_OCCUP,Scenario!="")
ICU_OCCUP <- subset(ICU_OCCUP,Scenario!="")
MEDSURG_OCCUP <- subset(MEDSURG_OCCUP,Scenario!="")

INPT_Melt <- melt(INPT_OCCUP,id=c("Open.Num","Trial.Num","Run.Num","Scenario","Type"),
                  value.name="Occupancy",variable.name="Week")
ICU_Melt <- melt(ICU_OCCUP,id=c("Open.Num","Trial.Num","Run.Num","Scenario","Type"),
                 value.name="Occupancy",variable.name="Week")
MS_Melt <- melt(MEDSURG_OCCUP,id=c("Open.Num","Trial.Num","Run.Num","Scenario","Type"),
                value.name="Occupancy",variable.name="Week")

INPT_Melt$Week <- as.numeric(gsub("Week.", "", INPT_Melt$Week))
ICU_Melt$Week <- as.numeric(gsub("Week.", "", ICU_Melt$Week))
MS_Melt$Week <- as.numeric(gsub("Week.", "", MS_Melt$Week))

INPT_Melt$Occupancy <- as.numeric(INPT_Melt$Occupancy)
ICU_Melt$Occupancy <- as.numeric(ICU_Melt$Occupancy)
MS_Melt$Occupancy <- as.numeric(MS_Melt$Occupancy)

INPT_Melt <- subset(INPT_Melt,!is.na(Occupancy))
ICU_Melt <- subset(ICU_Melt,!is.na(Occupancy))
MS_Melt <- subset(MS_Melt,!is.na(Occupancy))

scenario_list <- c("Status Quo","Schools Reopen","Schools Stay Closed","Aggressive")
ICU_Melt$Scenario <- factor(ICU_Melt$Scenario,ordered=TRUE,levels=scenario_list)
ICU_Melt$Type <- factor(ICU_Melt$Type,ordered=TRUE,levels=value_list)
MS_Melt$Scenario <- factor(MS_Melt$Scenario,ordered=TRUE,levels=scenario_list)
MS_Melt$Type <- factor(MS_Melt$Type,ordered=TRUE,levels=value_list)

ICU_Melt_Avg <- ddply(ICU_Melt,.(Scenario,Type,Week),summarize,Occupancy_Sm=mean(Occupancy))
MS_Melt_Avg <- ddply(MS_Melt,.(Scenario,Type,Week),summarize,Occupancy_Sm=mean(Occupancy))

ICU_Melt_Cast <- dcast(ICU_Melt_Avg, Scenario+Week~Type, mean)
MS_Melt_Cast <- dcast(MS_Melt_Avg, Scenario+Week~Type, mean)

ggplot(data=ICU_Melt_Avg,aes(x=Week,y=Occupancy_Sm,group=Type))+
  geom_line()+theme_bw()+facet_wrap(~Scenario) +
  xlab("Weeks Since Feb 21st") + ylab("Bed Occupancy") +
  labs(title="ICU Beds") 

ggplot(data=MS_Melt_Avg,aes(x=Week,y=Occupancy_Sm,group=Type))+
  geom_line()+theme_bw()+facet_wrap(~Scenario)+
  xlab("Weeks Since Feb 21st") + ylab("Bed Occupancy") +
  labs(title="Adult Acute Beds")

p1 <- ggplot(data=ICU_Melt_Cast,aes(x=Week,y=EV)) + 
  geom_ribbon(data=ICU_Melt_Cast,aes(xmin=1,xmax=7,ymin=LB,ymax=UB),fill="darkblue",alpha=.5) +
  geom_line(data=ICU_Melt_Cast,aes(x=Week,y=LB),color="darkgrey")+
  geom_line(data=ICU_Melt_Cast,aes(x=Week,y=EV),color="black")+
  geom_line(data=ICU_Melt_Cast,aes(x=Week,y=UB),color="darkgrey")+
  theme_bw()+facet_wrap(~Scenario) +
  xlab("Weeks Since Feb 21st") + ylab("Bed Occupancy") +
  labs(title="ICU Beds") 

p2 <- ggplot(data=MS_Melt_Cast,aes(x=Week,y=EV)) + 
  geom_ribbon(data=MS_Melt_Cast,aes(xmin=1,xmax=7,ymin=LB,ymax=UB),fill="darkgreen",alpha=.5) +
  geom_line(data=MS_Melt_Cast,aes(x=Week,y=LB),color="darkgrey")+
  geom_line(data=MS_Melt_Cast,aes(x=Week,y=EV),color="black")+
  geom_line(data=MS_Melt_Cast,aes(x=Week,y=UB),color="darkgrey")+
  theme_bw()+facet_wrap(~Scenario) +
  xlab("Weeks Since Feb 21st") + ylab("Bed Occupancy") +
  labs(title="Acute Adult Beds") 

write.csv(ICU_Melt_Cast,"ICU_Cast_Summary.csv",sep=",")
write.csv(MS_Melt_Cast,"MS_Cast_Summary.csv",sep=",")

pdf("Oregon_315_ICUandMSbeds.pdf",width=8,height=5)
plot(p1)
plot(p2)
dev.off()

