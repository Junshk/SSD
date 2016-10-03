require 'optim'
require 'make_net'
--require 'loss'
require 'gnuplot'
require 'FetchData'
require 'MultiBoxLoss'


Classes={'aeroplane','bicycle','bird','boat','bottle','bus','car',
           'cat','chair','cow','diningtable','dog','horse','motorbike',
           'person','pottedplant','sheep','sofa','train','tvmonitor'}

local confusion = optim.ConfusionMatrix(Classes)



--require 'prior_box'
--SGD
optimState ={

learningRate = 1e-3,
momentum = 0.9,
weightDecay = 0.0005

}

local batch_size =2 
-------------------------------------------------------------------------------

function training(opt)

local net = make_net('vgg')
net:training()
net:cuda()
cudnn.convert(net,cudnn)
--local criterion = nn.SSDloss(default)
--criterion:cuda()


local img_Info_table = ImgInfo()


local params, grads = net:getParameters()

local feval = function(x)

if x ~= params then
        params:copy(x)
        end



local input, target = patchFetch(batch_size,img_Info_table) --imgtensor, table
        
grads:zero()

local detc = torch.sum(torch.gt(target[1],21))+ torch.sum(torch.lt(target[1],1))
assert(detc~=0 , 'wrong class label')

local output = net:forward(input:cuda())
input:float()
-----------------------------------
print('forward')
local err, df_dx_loc,df_dx_conf = MultiBoxLoss(output,target)
------------------------------------
--local err = criterion:forward(output,target)
--local df_dx = criterion:backward(output,target)
print('loss')
--net:backward(input:cuda(),{df_dx_conf:cuda(),df_dx_loc:cuda()})
print('backward')


return err,grads
end -- end local feval


local losses = {}

for iteration =1,opt.end_iter do


if iteration == 40*1000 then optimState.learningRate = 1e-4 end


local _, loss = optim.sgd(feval,params,optimState)

table.insert(losses,loss[1])

if iteration%opt.plot_iter ==0 then
gnuplot.plot({'loss',torch.range(1,#losses),torch.Tensor(losses),'-'})
end

if iteration % opt.print_iter ==0 then 
        print('iter',iteration,'loss ',loss[1])
end


if iteration % opt.save_iter ==0 then 
        net:clearState()
        torch.save('model/'..basenet..'SSDnet_intm.t7')

end

end




net:clearState()
torch.save('model/'..basenet..'SSDnet.t7',net)


end
