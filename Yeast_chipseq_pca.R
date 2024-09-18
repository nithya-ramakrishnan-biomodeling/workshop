data_file="/Users/nitrama/My Drive/research/data/yeast_db.csv"
headers = c("H2AK5ac",   "H2AS129ph", "H3K14ac",  
            "H3K18ac",   "H3K23ac",   "H3K27ac" , 
             "H3K36me",   "H3K36me2",  "H3K36me3" ,
             "H3K4ac" ,   "H3K4me" ,"H3K4me2",  
             "H3K4me3",   "H3K56ac", "H3K79me" , 
             "H3K79me3",  "H3K9ac" ,  "H3S10ph",  
             "H4K12ac",   "H4K16ac",   "H4K20me",  
             "H4K5ac",    "H4K8ac",    "H4R3me" ,  
             "H4R3me2s",  "Htz1")
yeast_table1 = read.csv(data_file)
X = yeast_table1[headers]
yeast_genes = yeast_table1[,"gene"]
X1 = yeast_table1[(yeast_table1["gene"] != ""),steady_state_headers]

X1[is.na(X1)]=0
pcs = prcomp((X1))
pc1  = pcs$rotation[,1]
pc2 = pcs$rotation[,2]

plot(pc1, pc2, pch=16, ,cex=1.8,cex.axis=1.5,xlab="PC1",ylab="PC2",cex.lab=1.4)

text(pc1,pc2,labels=(headers))

kc <- kmeans(pcs$rotation[,1:2],centers = 2)
plot(pch=16,pcs$rotation[,1:2],col=factor(kc$cluster))
text(pc1,pc2,labels=(steady_state_headers),cex=0.5)

plot(yeast_table1["H3K36me3"]$H3K36me3,yeast_table1["H3K79me3"]$H3K79me3,xlab = "H3K36me3",ylab="H3K79me3")
print(cor(yeast_table1["H3K36me3"]$H3K36me3,yeast_table1["H3K79me3"]$H3K79me3))


plot(yeast_table1["H3K36me3"]$H3K36me3,yeast_table1["H3K56ac"]$H3K56ac,xlab = "H3K36me3",ylab="H3K56ac")
print(cor(yeast_table1["H3K36me3"]$H3K36me3,yeast_table1["H3K56ac"]$H3K56ac))
