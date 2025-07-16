require("stringr")
require("ape")

## Plot anomalous result 2018 #######################################################################################################################
h3n2_tree <- read.tree("projects/woodman/test_anomaly/H3N2_2018_data_2018_results/H3N2_simpleLabel.fasta.newick")
h3n2_tree <- ladderize(h3n2_tree,right = 'False')
# plot(h3n2_tree,show.tip.label = F)

ids <- read.table("projects/woodman/test_anomaly/H3N2_2018_data_2018_results/samplingYears")[,1]
samplingYear <- read.table("projects/woodman/test_anomaly/H3N2_2018_data_2018_results/samplingYears")[,2]
names(samplingYear) <- ids

colourForSamplingYear <- colorRampPalette(c("blue","orange"),space="Lab")(length(unique(samplingYear)))
names(colourForSamplingYear) <- sort(unique(samplingYear))

pdf(file = "projects/woodman/test_anomaly/H3N2_2018_data_2018_results/H3N2_anomaly.pdf",height=10,width=12)
plot(h3n2_tree,show.tip.label = F,edge.color = "lightgrey", main = "Result obtained in 2018",cex.main = 2)
plotinfo <- get("last_plot.phylo", envir = .PlotPhyloEnv)
# points(plotinfo$xx[1:length(h3n2_tree$tip.label)], plotinfo$yy[1:length(h3n2_tree$tip.label)],col="gray",pch=19,cex=0.66)
points(plotinfo$xx[1:length(h3n2_tree$tip.label)], plotinfo$yy[1:length(h3n2_tree$tip.label)],col=colourForSamplingYear[as.character(samplingYear[h3n2_tree$tip.label])],pch=19,cex=0.5)
add.scale.bar(x = 0.05,y=-200)
text(x = 0.126, y=-200,"subs/site")
legend("left",c("Sampling year",names(colourForSamplingYear)[seq(1,50,4)]),pch=19,col=c("white",colourForSamplingYear[seq(1,50,4)]),bty="n",cex = 2)
dev.off()

## Plot 5 attempts at reproduction ############################################################################################################################################

file_labels <- c("13","14_1","14_2","15_1","15_2","15_3")
loglik_list <- list()
loglik_list[["13"]] <- -170875
loglik_list[["14_1"]] <- -170843
loglik_list[["14_2"]] <- -170856
loglik_list[["15_1"]] <- -170917
loglik_list[["15_2"]] <- -170819
loglik_list[["15_3"]] <- -170905

ids <- read.table("projects/woodman/test_anomaly/H3N2_2018_data_2018_results/samplingYears")[,1]
samplingYear <- read.table("projects/woodman/test_anomaly/H3N2_2018_data_2018_results/samplingYears")[,2]
names(samplingYear) <- ids
colourForSamplingYear <- colorRampPalette(c("blue","red"),space="Lab")(length(unique(samplingYear)))
names(colourForSamplingYear) <- sort(unique(samplingYear))

fnames <- paste("projects/woodman/test_anomaly/H3N2_2018_data_2025_results/anomaly_July_",c("13","14_1","14_2","15_1","15_2","15_3"),".rooted_newick",sep="") 
# reproduction_file_names <- list()
h3n2_trees <- list()
for (i in c(1:6)){
  h3n2_trees[[file_labels[i]]] <- ladderize(read.tree(fnames[i]), right = 'False')
}


# Plot 6 figures
pdf(file = "projects/woodman/test_anomaly/H3N2_2018_data_2025_results/H3N2_6_reproductions.pdf",height=12,width=9)
par(mfrow=c(3,2), mar=c(2,2,3,1))

for (i in 1:6) {
  h3n2_tree <- h3n2_trees[[file_labels[i]]]
  loglik <- loglik_list[[file_labels[i]]]
  plot(h3n2_tree, show.tip.label = F, edge.color = "lightgrey", main = paste("log likelihood is", loglik))
  plotinfo <- get("last_plot.phylo", envir = .PlotPhyloEnv)
  points(plotinfo$xx[1:length(h3n2_tree$tip.label)], plotinfo$yy[1:length(h3n2_tree$tip.label)], col=colourForSamplingYear[as.character(samplingYear[h3n2_tree$tip.label])], pch=19, cex=0.3)
  add.scale.bar(x = 0.05, y=-200)
  if (i == 2) {
    text(x = 0.146, y=-200,"subs/site")
  } else if (floor(i/2) == ceiling(i/2)) {
    text(x = 0.136, y=-200,"subs/site")
  } else {
    text(x = 0.146, y=-200,"subs/site")
  }
  
  if (i == 3) {
    legend("left",c("Sampling year",names(colourForSamplingYear)[seq(1,50,4)]),pch=19,col=c("white",colourForSamplingYear[seq(1,50,4)]),bty="n")
  }
}

dev.off()


## Plot anomalous result reproduced on July 13 2025 #######################################################################################################################

h3n2_tree <- read.tree("projects/woodman/test_anomaly/H3N2_2018_data_2025_results/anomaly.rooted_newick")
h3n2_tree <- ladderize(h3n2_tree, right = 'False')

ids <- read.table("projects/woodman/test_anomaly/H3N2_2018_data_2018_results/samplingYears")[,1]
samplingYear <- read.table("projects/woodman/test_anomaly/H3N2_2018_data_2018_results/samplingYears")[,2]
names(samplingYear) <- ids
colourForSamplingYear <- colorRampPalette(c("blue","red"),space="Lab")(length(unique(samplingYear)))
names(colourForSamplingYear) <- sort(unique(samplingYear))


pdf(file = "projects/woodman/test_anomaly/H3N2_2018_data_2025_results/H3N2_July_13.pdf",height=9,width=12)
plot(h3n2_tree,show.tip.label = F,edge.color = "lightgrey", main = "Results reproduced in 2025")
plotinfo <- get("last_plot.phylo", envir = .PlotPhyloEnv)
points(plotinfo$xx[1:length(h3n2_tree$tip.label)], plotinfo$yy[1:length(h3n2_tree$tip.label)],col=colourForSamplingYear[as.character(samplingYear[h3n2_tree$tip.label])],pch=19,cex=0.3)
add.scale.bar(x = 0.05,y=-200)
text(x = 0.126, y=-200,"subs/site")
legend("left",c("Sampling year",names(colourForSamplingYear)[seq(1,50,4)]),pch=19,col=c("white",colourForSamplingYear[seq(1,50,4)]),bty="n")
dev.off()

## Plot anomalous result reproduced on July 14 I 2025 #######################################################################################################################

h3n2_tree <- read.tree("projects/woodman/test_anomaly/H3N2_2018_data_2025_results/anomaly_July_14.rooted_newick")
h3n2_tree <- ladderize(h3n2_tree, right = 'False')

# write.table(x = cbind(h3n2_full_tree$tip.label,distanceFromRoot),quote = F,row.names = F,col.names = c("id","distanceFromRoot"),sep="\t")
colourForSamplingYear <- colorRampPalette(c("blue","red"),space="Lab")(length(unique(samplingYear)))
names(colourForSamplingYear) <- sort(unique(samplingYear))


pdf(file = "projects/woodman/test_anomaly/H3N2_2018_data_2025_results/H3N2_July_14_1.pdf",height=9,width=12)
plot(h3n2_tree,show.tip.label = F,edge.color = "lightgrey", main = "Results reproduced in 2025")
plotinfo <- get("last_plot.phylo", envir = .PlotPhyloEnv)
# points(plotinfo$xx[1:length(h3n2_tree$tip.label)], plotinfo$yy[1:length(h3n2_tree$tip.label)],col="gray",pch=19,cex=0.66)
points(plotinfo$xx[1:length(h3n2_tree$tip.label)], plotinfo$yy[1:length(h3n2_tree$tip.label)],col=colourForSamplingYear[as.character(samplingYear[h3n2_tree$tip.label])],pch=19,cex=0.3)
add.scale.bar(x = 0.05,y=-200)
text(x = 0.126, y=-200,"subs/site")
legend("left",c("Sampling year",names(colourForSamplingYear)[seq(1,50,3)]),pch=19,col=c("white",colourForSamplingYear[seq(1,50,3)]),bty="n")
dev.off()

## Plot anomalous result reproduced on July 14 II 2025 #######################################################################################################################

h3n2_tree <- read.tree("projects/woodman/test_anomaly/H3N2_2018_data_2025_results/anomaly_July_14_2.rooted_newick")
h3n2_tree <- ladderize(h3n2_tree, right = 'False')

# write.table(x = cbind(h3n2_full_tree$tip.label,distanceFromRoot),quote = F,row.names = F,col.names = c("id","distanceFromRoot"),sep="\t")
colourForSamplingYear <- colorRampPalette(c("blue","red"),space="Lab")(length(unique(samplingYear)))
names(colourForSamplingYear) <- sort(unique(samplingYear))

pdf(file = "projects/woodman/test_anomaly/H3N2_2018_data_2025_results/H3N2_July_14_2.pdf",height=9,width=12)
plot(h3n2_tree,show.tip.label = F,edge.color = "lightgrey", main = "Results reproduced in 2025")
plotinfo <- get("last_plot.phylo", envir = .PlotPhyloEnv)
# points(plotinfo$xx[1:length(h3n2_tree$tip.label)], plotinfo$yy[1:length(h3n2_tree$tip.label)],col="gray",pch=19,cex=0.66)
points(plotinfo$xx[1:length(h3n2_tree$tip.label)], plotinfo$yy[1:length(h3n2_tree$tip.label)],col=colourForSamplingYear[as.character(samplingYear[h3n2_tree$tip.label])],pch=19,cex=0.3)
add.scale.bar(x = 0.05,y=-200)
text(x = 0.126, y=-200,"subs/site")
legend("left",c("Sampling year",names(colourForSamplingYear)[seq(1,50,3)]),pch=19,col=c("white",colourForSamplingYear[seq(1,50,3)]),bty="n")
dev.off()

## Plot anomalous result reproduced on July 15 I 2025 #######################################################################################################################

h3n2_tree <- read.tree("projects/woodman/test_anomaly/H3N2_2018_data_2025_results/anomaly_July_15_1.rooted_newick")
h3n2_tree <- ladderize(h3n2_tree, right = 'False')

# write.table(x = cbind(h3n2_full_tree$tip.label,distanceFromRoot),quote = F,row.names = F,col.names = c("id","distanceFromRoot"),sep="\t")
colourForSamplingYear <- colorRampPalette(c("blue","red"),space="Lab")(length(unique(samplingYear)))
names(colourForSamplingYear) <- sort(unique(samplingYear))

pdf(file = "projects/woodman/test_anomaly/H3N2_2018_data_2025_results/H3N2_July_15_1.pdf",height=9,width=12)
plot(h3n2_tree,show.tip.label = F,edge.color = "lightgrey", main = "Results reproduced in 2025")
plotinfo <- get("last_plot.phylo", envir = .PlotPhyloEnv)
# points(plotinfo$xx[1:length(h3n2_tree$tip.label)], plotinfo$yy[1:length(h3n2_tree$tip.label)],col="gray",pch=19,cex=0.66)
points(plotinfo$xx[1:length(h3n2_tree$tip.label)], plotinfo$yy[1:length(h3n2_tree$tip.label)],col=colourForSamplingYear[as.character(samplingYear[h3n2_tree$tip.label])],pch=19,cex=0.3)
add.scale.bar(x = 0.05,y=-200)
text(x = 0.126, y=-200,"subs/site")
legend("right",c("Sampling year",names(colourForSamplingYear)[seq(1,50,3)]),pch=19,col=c("white",colourForSamplingYear[seq(1,50,3)]),bty="n")
dev.off()

## Plot anomalous result reproduced on July 15 II 2025 #######################################################################################################################

h3n2_tree <- read.tree("projects/woodman/test_anomaly/H3N2_2018_data_2025_results/anomaly_July_15_2.rooted_newick")
h3n2_tree <- ladderize(h3n2_tree,right = 'False')

# write.table(x = cbind(h3n2_full_tree$tip.label,distanceFromRoot),quote = F,row.names = F,col.names = c("id","distanceFromRoot"),sep="\t")
colourForSamplingYear <- colorRampPalette(c("blue","red"),space="Lab")(length(unique(samplingYear)))
names(colourForSamplingYear) <- sort(unique(samplingYear))

pdf(file = "projects/woodman/test_anomaly/H3N2_2018_data_2025_results/H3N2_July_15_2.pdf",height=9,width=12)
plot(h3n2_tree,show.tip.label = F,edge.color = "lightgrey", main = "Results reproduced in 2025")
plotinfo <- get("last_plot.phylo", envir = .PlotPhyloEnv)
# points(plotinfo$xx[1:length(h3n2_tree$tip.label)], plotinfo$yy[1:length(h3n2_tree$tip.label)],col="gray",pch=19,cex=0.66)
points(plotinfo$xx[1:length(h3n2_tree$tip.label)], plotinfo$yy[1:length(h3n2_tree$tip.label)],col=colourForSamplingYear[as.character(samplingYear[h3n2_tree$tip.label])],pch=19,cex=0.3)
add.scale.bar(x = 0.05,y=-200)
text(x = 0.126, y=-200,"subs/site")
legend("right",c("Sampling year",names(colourForSamplingYear)[seq(1,50,3)]),pch=19,col=c("white",colourForSamplingYear[seq(1,50,3)]),bty="n")
dev.off()

## Plot anomalous result reproduced on July 15 III 2025 #######################################################################################################################

h3n2_tree <- read.tree("projects/woodman/test_anomaly/H3N2_2018_data_2025_results/anomaly_July_15_3.rooted_newick")
h3n2_tree <- ladderize(h3n2_tree, right = 'False')

# write.table(x = cbind(h3n2_full_tree$tip.label,distanceFromRoot),quote = F,row.names = F,col.names = c("id","distanceFromRoot"),sep="\t")
colourForSamplingYear <- colorRampPalette(c("blue","red"),space="Lab")(length(unique(samplingYear)))
names(colourForSamplingYear) <- sort(unique(samplingYear))

pdf(file = "projects/woodman/test_anomaly/H3N2_2018_data_2025_results/H3N2_July_15_3.pdf",height=9,width=12)
plot(h3n2_tree,show.tip.label = F,edge.color = "lightgrey", main = "Results reproduced in 2025")
plotinfo <- get("last_plot.phylo", envir = .PlotPhyloEnv)
# points(plotinfo$xx[1:length(h3n2_tree$tip.label)], plotinfo$yy[1:length(h3n2_tree$tip.label)],col="gray",pch=19,cex=0.66)
points(plotinfo$xx[1:length(h3n2_tree$tip.label)], plotinfo$yy[1:length(h3n2_tree$tip.label)],col=colourForSamplingYear[as.character(samplingYear[h3n2_tree$tip.label])],pch=19,cex=0.3)
add.scale.bar(x = 0.05,y=-200)
text(x = 0.126, y=-200,"subs/site")
legend("left",c("Sampling year",names(colourForSamplingYear)[seq(1,50,3)]),pch=19,col=c("white",colourForSamplingYear[seq(1,50,3)]),bty="n")
dev.off()


