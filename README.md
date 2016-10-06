# SSD
Wei Liu et al. SSD: Single Shot Multibox Detector.

torch ver.

# code desciption

ChannelNormalization : L2normalization layer. normalize each feature.

FetchData : 

Mul_modified : scale layer. not channel shared parameters.

MultiBoxLoss : SSD loss (loc and conf).

dataload : download pascal data 2007 trainval, 2012 train.

pascal : pascal data parsing and load pascal image list.

make_net : make 500 SSD model.

prior_box :

etc :
