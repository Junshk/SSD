require 'nn'
require 'cudnn'
require 'cunn'
require 'loadcaffe'
require 'modules/ChannelNormalization'
require 'modules/Mul_modified'

local nninit = require 'nninit'
local net = nn.Sequential()

-------------------------------------------------------------------------------
local function base_load(base_name)

local base_net
local vgg_path, resnet_path = './VGG_ILSVRC_16_layers.caffemodel', './'


if base_name == 'vgg' and paths.filep(vgg_path) ~= true then
os.execute('wget http://www.robots.ox.ac.uk/~vgg/software/very_deep/caffe/VGG_ILSVRC_16_layers.caffemodel')
elseif base_name == 'residual' and paths.filep(resnet_path) ~= true then
local url = 'https://onedrive.live.com/?authkey=%21AAFW2-FVoxeVRck&id=4006CBB8476FF777%2117887&cid=4006CBB8476FF777'
os.execute('wget '..url..'/ResNet_mean.binaryproto')
os.execute('wget '..url..'/ResNet-50-deploy.prototxt')
os.execute('wget '..url..'/ResNet-50-modle.caffemodel')

end
  
if base_name == 'vgg' then 
base_net = loadcaffe.load('VGG_ILSVRC_layers_deploy.prototxt','VGG_ILSVRC_16_layers.caffemodel','cudnn')
elseif base_name == 'residual' then
base_net = loadcaffe.load()
else assert(false,'wrong base network name')
end

return base_net
end

-------------------------------------------------------------------------------

function loc() --table input, linear output
local locnet = nn.Sequential()
local parl =nn.ParallelTable()
local reshape = nn.ParallelTable()
  
local dim ={256,256,256,256,512,1024,512}
  
for iter_loc = 1, 6 do 
parl:add(nn.SpatialConvolution(dim[iter_loc],4*6,3,3,1,1,1,1):init('weight',nninit.xavier))
end
parl:add(nn.SpatialConvolution(dim[7],4*3,3,3,1,1,1,1):init('weight',nninit.xavier))

reshape:add(nn.Reshape(4,6*1))
reshape:add(nn.Reshape(4,6*2*2))
reshape:add(nn.Reshape(4,6*4*4))
reshape:add(nn.Reshape(4,6*8*8))
reshape:add(nn.Reshape(4,6*16*16))
reshape:add(nn.Reshape(4,6*32*32))
reshape:add(nn.Reshape(4,3*63*63))
  
  
  locnet:add(parl)
  locnet:add(reshape)
  locnet:add(nn.JoinTable(2,2))

--locnet:add(nn.FlattenTable())
cudnn.convert(locnet,cudnn)
return locnet
end

function conf(classes)
local dim ={256,256,256,256,512,1024,512}
local parl =nn.ParallelTable()
local reshape = nn.ParallelTable()
local confnet = nn.Sequential()
  
for iter_conf = 1, 6 do 
parl:add(nn.SpatialConvolution(dim[iter_conf],classes*6,3,3,1,1,1,1):init('weight',nninit.xavier))
end
parl:add(nn.SpatialConvolution(dim[7],classes*3,3,3,1,1,1,1):init('weight',nninit.xavier))

reshape:add(nn.Reshape(classes,6*1))
reshape:add(nn.Reshape(classes,6*2*2))
reshape:add(nn.Reshape(classes,6*4*4))
reshape:add(nn.Reshape(classes,6*8*8))
reshape:add(nn.Reshape(classes,6*16*16))
reshape:add(nn.Reshape(classes,6*32*32))
reshape:add(nn.Reshape(classes,3*63*63))

  confnet:add(parl)
  confnet:add(reshape)
  confnet:add(nn.JoinTable(2,2))
--confnet:add(nn.FlattenTable())
cudnn.convert(confnet,cudnn)
return confnet
end


-------------------------------------------------------------------------------

function make_net(base_name)


local base = base_load(base_name)
base:float() ; cudnn.convert(base,nn)
-- fc 6, 7 to conv and subsampling parameters
local weight_of_fc6 = base.modules[33].weight:reshape(4096,7,7,512)
local perm_order = torch.randperm(4096)
perm_order = perm_order[{{1,1024}}]
local sample = torch.Tensor({1,4,7}):long()


weight_of_fc6 = weight_of_fc6:index(1,perm_order:long())
weight_of_fc6 = weight_of_fc6:index(2,sample)
weight_of_fc6 = weight_of_fc6:index(3,sample)

weight_of_fc6 = weight_of_fc6:transpose(3,4)
weight_of_fc6 = weight_of_fc6:transpose(2,3)
weight_of_fc6 = weight_of_fc6:transpose(1,2)

local weight_of_fc7 = base.modules[36].weight:reshape(4096,1024,4,1)
weight_of_fc7 = weight_of_fc7:index(1,perm_order:long())
weight_of_fc7 = weight_of_fc7[{{},{},{1},{}}]
weight_of_fc7 = weight_of_fc7:transpose(1,2)

-----------------------------

if base_name == 'vgg' then


local classes = 21
local extra = nn.Sequential()  -- after pool5
local concat1,concat2,concat3,concat4,concat5 = nn.ConcatTable(),nn.ConcatTable(),nn.ConcatTable(),nn.ConcatTable(),nn.ConcatTable()
local seq1,seq2,seq3,seq4 = nn.Sequential(),nn.Sequential(), nn.Sequential(), nn.Sequential()

local seq5, concat6 = nn.Sequential(), nn.ConcatTable()

concat6:add(cudnn.SpatialAveragePooling(2,2))
concat6:add(nn.Identity())

seq5:add(cudnn.SpatialConvolution(256,128,1,1):init('weight',nninit.xavier))
seq5:add(cudnn.ReLU(true))
seq5:add(cudnn.SpatialConvolution(128,256,3,3,2,2,1,1):init('weight',nninit.xavier))
seq5:add(cudnn.ReLU(true))
seq5:add(concat6)
concat5:add(seq5)
concat5:add(nn.Identity())


seq4:add(cudnn.SpatialConvolution(256,128,1,1):init('weight',nninit.xavier))
seq4:add(cudnn.ReLU(true))
seq4:add(cudnn.SpatialConvolution(128,256,3,3,2,2,1,1):init('weight',nninit.xavier))
seq4:add(cudnn.ReLU(true))
seq4:add(concat5)
concat4:add(seq4)
concat4:add(nn.Identity())

seq3:add(cudnn.SpatialConvolution(512,128,1,1):init('weight',nninit.xavier))
seq3:add(cudnn.ReLU(true))
seq3:add(cudnn.SpatialConvolution(128,256,3,3,2,2,1,1):init('weight',nninit.xavier))
seq3:add(cudnn.ReLU(true))
seq3:add(concat4)
concat3:add(seq3)
concat3:add(nn.Identity())

seq2:add(cudnn.SpatialConvolution(1024,256,1,1):init('weight',nninit.xavier))
seq2:add(cudnn.ReLU(true))
seq2:add(cudnn.SpatialConvolution(256,512,3,3,2,2,1,1):init('weight',nninit.xavier))
seq2:add(cudnn.ReLU(true))
seq2:add(concat3)
concat2:add(seq2)
concat2:add(nn.Identity())--nn.SpatialConvolution(1024,6*(classes+4),3,3,1,1,1,1)) --classifier



for iter = 24, 30 do
seq1:add(base.modules[iter])
end
seq1:add(cudnn.SpatialMaxPooling(3,3,1,1,1,1))

seq1:add(nn.SpatialDilatedConvolution(512,1024,3,3,1,1,6,6,6,6):init('weight',nninit.copy,weight_of_fc6))  -- subsampling fc 6
seq1:add(cudnn.SpatialConvolution(1024,1024,1,1):init('weight',nninit.copy,weight_of_fc7)) -- subsampling fc 7
seq1:add(concat2)

concat1:add(seq1)
local ss = nn.Sequential():add(nn.ChannelNormalization(2))
ss:add(nn.Mul_modified(512,20))--nn.Mul():init('weight',nninit.constant,20))
concat1:add(ss)--cudnn.SpatialConvolution(512,3*(classes+4),3,3,1,1,1,1))
-- 4_3

for iter = 1, 23 do
net:add(base.modules[iter])
end
net:add(concat1)

net:add(nn.FlattenTable())


-- here is loc prior , conf
local loss_net = nn.Sequential()

local concat = nn.ConcatTable()
local loc_net = loc()
local conf_net =conf(classes)



--------------------------------
------------------------------
concat:add(conf_net)
concat:add(loc_net)
loss_net:add(concat)
loss_net:add(nn.JoinTable(2,1))

net:add(loss_net) -- 2 table output
elseif base_name == 'residual' then

print('residual')
else assert(false,'wrong base network name')
end

base =nil; collectgarbage();
net:cuda()

return net

end
cudnn.fastest = true
