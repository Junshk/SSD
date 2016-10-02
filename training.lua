require 'optim'
require 'make_net'
require 'loss'
require 'gnuplot'
require 'FetchData'
require 'MultiBoxLoss'


Classes={'aeroplane','bicycle','bird','boat','bottle','bus','car',
           'cat','chair','cow','diningtable','dog','horse','motorbike',
           'person','pottedplant','sheep','sofa','train','tvmonitor'}





--require 'prior_box'
--SGD
optimState ={

learningRate = 1e-3,
momentum = 0.9,
weightDecay = 0.0005

}

local batch_size = 32
-------------------------------------------------------------------------------

function training()

local net = make_net('vgg')
net:training()
net:cuda()
cudnn.convert(net,cudnn)
local criterion = nn.SSDloss(default)
--criterion:cuda()


local img_Info_table = ImgInfo()


local params, grads = net:getParameters()

local feval = function(x)

if x ~= params then
        params:copy(x)
        end



local input, target = patchFetch(batch_size,img_Info_table) --imgtensor, table
        
grads:zero()



local output = net:forward(input:cuda())
-----------------------------------

local err, df_dx = MultiBoxLoss(output,target)
------------------------------------
--local err = criterion:forward(output,target)
--local df_dx = criterion:backward(output,target)

net:backward(input:cuda(),df_dx)



return err,grads
end -- end local feval


local losses = {}

for iteraition =1,end_iter do

local _, loss = optim.sgd(feval,params,optimState)

table.insert(losses,loss[1])

if iteraition%plot_iter ==0 then
gnuplot.plot({})
end



end




net:clearState()
torch.save('model/'..basenet..'SSDnet.t7',net)


end
