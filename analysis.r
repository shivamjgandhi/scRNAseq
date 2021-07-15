singlecell()

# load data
# ---------
counts = read_mtx('~/smillie-data/data/scbac/BacDrop-Hung/proc/all')
g.info = ffread('~/smillie-data/data/scbac/BacDrop-Hung/ref/GCF_000694555.gene_map.txt', sep='\t', header=T, as.dt=T)
g.info = g.info[order(contig, beg)]
g.info = g.info[!duplicated(g.info$id)]
samples = c('P1', 'P2', 'P3', 'P4')

# fix gene order
# --------------
i = unique(rownames(counts)[match(g.info$id, rownames(counts))])
counts = as(counts[na.omit(i),], 'sparseMatrix')

# define operons
# --------------
m = nrow(g.info)
d = g.info$beg - c(0, g.info$end[1:(m-1)]) > 100
operons = cumsum(d)
o2g = tapply(g.info$id, operons, function(a) sort(unique(a)))
g2o = tapply(operons, g.info$id, function(a) sort(unique(a)))

# operon description
# ------------------
setkey(g.info, id)
o2d = sapply(o2g, function(a){b = table(substr(g.info[a]$gene, 1, 3)); paste(names(sort(b, dec=T))[[1]], max(b), sep='.')})
i = grepl('^\\.', o2d)
o2d[i] = names(o2d)[i]

# subset counts
# -------------
i = rowSums(counts > 0) >= 50
j = colSums(counts > 0) >= 50
counts = counts[i, j]
counts.op = nice_agg(counts, g2o[rownames(counts)], 'sum')

# single cell analysis
# --------------------
sco = run_seurat(name='all', counts=counts, ming=0, minc=0, num_pcs=25, write_out=F)
sco.ps = sapply(samples, function(a) run_seurat(name=a, counts=counts[,grepl(paste0('^', a), colnames(counts))], ming=0, minc=0, num_pcs=25, write_out=F), simplify=F)
sco.op = run_seurat(name='op', counts=counts.op, ming=0, minc=0, num_pcs=25, write_out=F)

# cluster cells
# -------------
v = run_graph_cluster(sco$pca.rot[,1:25], k=100)
v.ps = sapply(sco.ps, function(a) run_graph_cluster(a$pca.rot[,1:25], k=100), simplify=F)
v.op = run_graph_cluster(sco.op$pca.rot[,1:25], k=100)
