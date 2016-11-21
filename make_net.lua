--require 'nn'
require 'cudnn'
require 'cunn'
require 'loadcaffe'
require 'modules/ChannelNormalization'
require 'nnlr'


cudnn.fastest = true
cudnn.benchmark = true

local nninit = require 'nninit'

local function ConvInit(dim1,dim2,k,s,p)
local k = k or 3
local s = s or 1
local p = p or math.floor(k/2)
return nn.SpatialConvolution(dim1,dim2,k,k,s,s,p,p):init('weight',nninit.xavier):init('bias',nninit.constant,0):learningRate('weight',1):learningRate('bias',2):weightDecay('weight',1):weightDecay('bias',0)
end
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
base_net = torch.load('VGG16.net')--loadcaffe.load('VGG_ILSVRC_layers_deploy.prototxt','VGG_ILSVRC_16_layers.caffemodel','nn')
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
parl:add(ConvInit(dim[iter_loc],4*6))
end
parl:add(ConvInit(dim[7],4*3))

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

return locnet
end

function conf(classes)
local dim ={256,256,256,256,512,1024,512}
local parl =nn.ParallelTable()
local reshape = nn.ParallelTable()
local confnet = nn.Sequential()
  
for iter_conf = 1, 6 do 
parl:add(ConvInit(dim[iter_conf],classes*6))
end
parl:add(ConvInit(dim[7],classes*3))

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

return confnet
end

function pretrain2(base)
local seq = nn.Sequential()
local n = #base.modules

for iter = 24, n do
 if iter <=30 then
   if base.modules[iter].weight ~=nil then
    seq:add(base.modules[iter]:learningRate('weight',1):learningRate('bias',2):weightDecay('weight',1):weightDecay('bias',0))
   else  
 seq:add( base.modules[iter])
    end
  end
end
return seq
end
function pretrain1(base)
local seq = nn.Sequential()
-- remove
local n = #base.modules

for iter = 9, 23 do 


--if iter <=23 then
    if base.modules[iter].weight ~= nil then
      base.modules[iter]:learningRate('weight',1):learningRate('bias',2):weightDecay('weight',1):weightDecay('bias',0)  
    end
--  end

  seq:add(base.modules[iter])
--cal conv = base.modules[iter]:clone('weight','bias')
  end

return seq
end
function pretrain0(base)
local seq = nn.Sequential()
for iter = 1, 8 do
--if  iter <=8 then 
    if base.modules[iter].weight ~= nil then 
      base.modules[iter]:learningRate('weight',0):learningRate('bias',0):weightDecay('weight',0):weightDecay('bias',0)
      
    end
    seq:add(base.modules[iter])
end
seq.accGradParameters = function() end

return seq
end
-------------------------------------------------------------------------------

function make_net(ch,mul)
ch = ch or false
mul = mul or false
local net = nn.Sequential()
local base_name = 'vgg'
local base = base_load(base_name)
--base:float() ; 
--base.accGradParameters = function() end
-- fc 6, 7 to conv and subsampling parameters
local weight_of_fc6 = base.modules[33].weight:reshape(4096,7,7,512)
local bias_of_fc6 = base.modules[33].bias

local perm_order = torch.randperm(4096)
perm_order = perm_order[{{1,1024}}]
local sample = torch.Tensor({2,4,6}):long()


weight_of_fc6 = weight_of_fc6:index(1,perm_order:long())
weight_of_fc6 = weight_of_fc6:index(2,sample)
weight_of_fc6 = weight_of_fc6:index(3,sample)

weight_of_fc6 = weight_of_fc6:transpose(3,4)
weight_of_fc6 = weight_of_fc6:transpose(2,3)
weight_of_fc6 = weight_of_fc6:transpose(1,2)

bias_of_fc6 = bias_of_fc6:index(1,perm_order:long())

local weight_of_fc7 = base.modules[36].weight:reshape(4096,1024,4,1)
weight_of_fc7 = weight_of_fc7:index(1,perm_order:long())
weight_of_fc7 = weight_of_fc7[{{},{},{1},{}}]
weight_of_fc7 = weight_of_fc7:transpose(1,2)

local bias_of_fc7 = base.modules[36].bias:reshape(1024,4,1)
--bias_of_fc7 = bias_of_fc7:index(1,perm_order:long())
bias_of_fc7 = bias_of_fc7[{{},{1},{}}]

bias_of_fc7 = bias_of_fc7:squeeze()
-----------------------------

if base_name == 'vgg' then


local classes = 21
local extra = nn.Sequential()  -- after pool5
local concat1,concat2,concat3,concat4,concat5 = nn.ConcatTable(),nn.ConcatTable(),nn.ConcatTable(),nn.ConcatTable(),nn.ConcatTable()
local seq1,seq2,seq3,seq4 = nn.Sequential(),nn.Sequential(), nn.Sequential(), nn.Sequential()

local seq5, concat6 = nn.Sequential(), nn.ConcatTable()

concat6:add(nn.SpatialAveragePooling(2,2))
concat6:add(nn.Identity())

seq5:add(ConvInit(256,128,1))
seq5:add(nn.ReLU(true))
seq5:add(ConvInit(128,256,3,2))
seq5:add(nn.ReLU(true))
seq5:add(concat6)
concat5:add(seq5)
concat5:add(nn.Identity())


seq4:add(ConvInit(256,128,1))
seq4:add(nn.ReLU(true))
seq4:add(ConvInit(128,256,3,2))
seq4:add(nn.ReLU(true))
seq4:add(concat5)
concat4:add(seq4)
concat4:add(nn.Identity())

seq3:add(ConvInit(512,128,1))
seq3:add(nn.ReLU(true))
seq3:add(ConvInit(128,256,3,2))
seq3:add(nn.ReLU(true))
seq3:add(concat4)
concat3:add(seq3)
concat3:add(nn.Identity())

seq2:add(ConvInit(1024,256,1))
seq2:add(nn.ReLU(true))
seq2:add(ConvInit(256,512,3,2))
seq2:add(nn.ReLU(true))
seq2:add(concat3)
concat2:add(seq2)
concat2:add(nn.Identity())--nn.SpatialConvolution(1024,6*(classes+4),3,3,1,1,1,1)) --classifier


--[[
for iter = 24, 30 do
seq1:add(base.modules[iter])
end]]--

seq1:add(pretrain2(base))
seq1:add(nn.SpatialMaxPooling(3,3,1,1,1,1))
--bias_of_fc6:fill(0)
--bias_of_fc7:fill(0)
seq1:add(nn.SpatialDilatedConvolution(512,1024,3,3,1,1,6,6,6,6):init('weight',nninit.copy,weight_of_fc6):init('bias',nninit.copy,bias_of_fc6):learningRate('weight',1):weightDecay('weight',1):learningRate('bias',2):weightDecay('bias',0))  -- subsampling fc 6
seq1:add(nn.ReLU(true))
seq1:add(nn.SpatialConvolution(1024,1024,1,1):init('weight',nninit.copy,weight_of_fc7):init('bias',nninit.copy,bias_of_fc7):learningRate('bias',2):weightDecay('bias',0):learningRate('weight',1):weightDecay('weight',1)) -- subsampling fc 7
seq1:add(nn.ReLU(true))
seq1:add(concat2)

concat1:add(seq1)
local ss = nn.Sequential()
local cmul = nn.CMul(1,512,1,1):init('weight',nninit.constant,20)
if ch==true then ss:add(nn.ChannelNormalization(2)) end
if mul==true then ss:add(cmul) end

concat1:add(ss)
net:add(pretrain1(base))
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

 collectgarbage();
net = cudnn.convert(net,cudnn):cuda()

return net, pretrain0(base)

end
--cudnn.fastest = true

