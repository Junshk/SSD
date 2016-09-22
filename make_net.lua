require 'nn'
require 'cudnn'
require 'cunn'
require 'loadcaffe'

local net = nn.Sequential()


local function base_load(base_name)

local base_net
local vgg_path, resnet_path = '', ''


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

function make_net(base_name)


base_load(base_name)


if base_name == 'vgg' then


local classes = 20
local extra = nn.Sequential()  -- after pool5
local concat1,concat2,concat3,concat4,concat5 = nn.ConcatTable(),nn.ConcatTable(),nn.ConcatTable(),nn.ConcatTable(),nn.ConcatTable()
local seq1,seq2,seq3,seq4 = nn.Sequential(),nn.Sequential(), nn.Sequential(), nn.Sequential()

concat5:add(nn.SpatialAveragePooling(3,3))
concat5:add(nn.Identity())

seq4:add(nn.SpatialConvolution(256,128,1,1))
seq4:add(nn.SpatialConvolution(128,256,3,3,2,2,1,1))
seq4:add(concat5)
concat4:add(seq4)
concat4:add(nn.Identity())

seq3:add(nn.SpatialConvolution(512,128,1,1))
seq3:add(nn.SpatialConvolution(128,256,3,3,2,2,1,1))
seq3:add(concat4)
concat3:add(seq3)
concat3:add(nn.Identity())

seq2:add(nn.SpatialConvolution(1024,256,1,1))
seq2:add(nn.SpatialConvolution(256,512,3,3,2,2,1,1))
seq2:add(concat3)
concat2:add(seq2)
concat2:add(nn.SpatialConvolution(1024,6*(classes+4),3,3,1,1,1,1)) --classifier

seq1:add(nn.SpatialConvolution(512,1024,3,3,1,1))  -- subsampling
seq1:add(nn.SpatialConvolution(1024,1024,1,1)) -- subsampling
seq1:add()



concat1:add(nn.SpatialConvolution(512,3*(classes+4),3,3,1,1,1,1)) -- classifier for conv4_3
concat1:add(nn.SpatialConvolution())


-- base network
-- fc 6, 7 to conv and subsampling parameters


-- remove dropout , fc8
base:remove(40)
base:remove(39)
base:remove(38)
base:remove(36)
base:remove(35)
base:remove(33)
-- pool 5 2 by 2 to 3 by 3
base.modules[31] = cudnn.SpatialMaxPooling(3,3,1,1)
-- atrous?




net:add(base)
net:add(extra)


elseif base_name == 'residual' then


else assert(false,'wrong base network name')
end



cudnn.convert(net,cudnn)
return net

end
