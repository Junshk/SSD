require 'option'
dofile('dataload.lua')

print('load datas')

---- pre-setting
if paths.filep('VGG16.net') ==false then 
  dofile('utils/caffe.lua')
end

losses = {}
val_losses = {}
accuracies = {}
start_iter = 1

local opt = Option
	
if paths.dirp('model') ==false then os.execute('mkdir model') end
 
netname = opt.netname

  require 'make_net'
  net =  make_net(opt.ch,opt.mul)
  pretrain = torch.load('pretrain.net')
  pretrain:evaluate()
  pretrain:clearState() 


if paths.filep('model/'..netname..'_intm.net') == false then 
opt.cont =false end

  if opt.cont == true then
    losses = torch.load('model/lossof'..netname..'_intm.t7')
    start_iter = #losses
    net = torch.load('model/'..netname..'_intm.net') 
    accuracies = torch.load('model/accof'..netname..'_intm.t7')
    privOpt = torch.load('model/optof'..netname..'.t7')
    print('privious opt',privOpt)
  
    while #accuracies>= start_iter do
    table.remove(accuracies,#accuracies)
    end
 end
-- net = cudnn.convert(net,cudnn)
-- net.modules[3].modules[2].modules[1] = cudnn.convert(net.modules[3].modules[2].modules[1],nn)
--print(net.modules[3].modules[2].modules[1])
----------------
--require 'training'
require 'Train2'
require 'donkey'
net:training()
training()

-- test code
require 'test'
local test_txt = 'VOCdevkit/VOC2012_test/ImageSets/Main/test.txt'
local test_list = {}
local f_test = assert(io.open(test_txt,"r"))

io.input(f_test)

while true do
  line = io.read() 
  if line == nil then break end
  table.insert(test_list,line)
end
f_test:close()

local test_folder = 'Test/'
local net = torch.load(netname..'.net')

test(net,test_list,test_folder)

os.execute('matlab -r -nodisplay plot_map')
