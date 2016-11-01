require 'optim'
require 'make_net'

require 'gnuplot'
require 'FetchData'
require 'MultiBoxLoss'
require 'test'
cutorch.setDevice(1)

--require 'prior_box'
--SGD
optimState ={

learningRate = 1e-3,
momentum = 0.9,
weightDecay = 0.0005

}

--local batch_size = 15 
-------------------------------------------------------------------------------

function training(opt)
local batch_size = opt.batch_size
local multi_batch = opt.multi_batch 
--local basenet = 'vgg'
if paths.dirp('model') ==false then os.execute('mkdir model') end

local net 
local netname = opt.netname--basenet .. '_b'.. batch_size

--cudnn.convert(net,cudnn)
local losses = {}
local val_losses = {}
local start_iter = 1

if paths.filep('model/'..netname..'_intm.net') == false then 
opt.cont =false end

  if opt.cont == true then
    losses = torch.load('model/lossof'..netname..'_intm.t7')
    start_iter = #losses
    net = torch.load('model/'..netname..'_intm.net')  
    privOpt = torch.load('model/optof'..netname..'.t7')
    print('privious opt',privOpt)
  else net = make_net(opt.ch,opt.mul) end

print(opt)
net:training()
net:cuda()
print(net)
print('training')

local img_Info_table = ImgInfo()

local params, grads = net:getParameters()

local feval = function(x)

if x ~= params then
        params:copy(x)
        end



grads:zero()
--net:clearState()
--local detc = torch.sum(torch.gt(target[1],21))+ torch.sum(torch.lt(target[1],1))
--assert(detc==0 , 'wrong class label')
local f =0

for iter = 1, multi_batch do

local input, target = patchFetch(batch_size,img_Info_table) --imgtensor, table
--print(cutorch.getMemoryUsage(1))

local output = net:forward(input:cuda())
input:float()
--print(cutorch.getMemoryUsage(1))
-----------------------------------
--net:float()
local err, df_dx = MultiBoxLoss(output:float(),target,opt.lambda)
f = f+ err
------------------------------------
--net:cuda()
--print(cutorch.getMemoryUsage(1))

net:backward(input:cuda(),df_dx:cuda())
--print(cutorch.getMemoryUsage(1))
end
--input = nil; df_dx = nil;
--target =nil;

f = f/ multi_batch
grads:div(multi_batch)

collectgarbage();
return f, grads
end -- end local feval
-------------------------------------------


for iteration = start_iter,opt.end_iter do
  
  if iteration == 60*1000 then optimState.learningRate = 1e-4 end
  local _, loss = optim.sgd(feval,params,optimState)
  table.insert(losses,loss[1])

  if opt.valid ==true and iteration%opt.test_iter ==0 then
         validation(net,'valid_loss_'..iteration,option.netname)   
  end

  if iteration % opt.print_iter ==0 then 
        print('iter',iteration,'loss ',loss[1])
  end

  if iteration%opt.plot_iter ==0 then
    local start_num, end_num = 
    math.max(1,iteration-opt.plot_iter*100), iteration
    gnuplot.plot({netname..'loss',torch.range(start_num,end_num),torch.Tensor(losses)[{{start_num,end_num}}],'-'})
  end

  
  if iteration % opt.save_iter ==0 then 
        net:clearState()
--        net:float()
--        cudnn.convert(net,nn)
        print('net saving')
        torch.save('model/'..netname..'_intm.net',net)

        torch.save('model/lossof'..netname..'_intm.t7',losses)
        torch.save('model/optof'..netname..'.t7',opt)
        print('net saved')
--        net = cudnn.convert(net,cudnn):cuda()
   end

end




  net:clearState()


  torch.save('model/'..netname..'.net',net)
  torch.save('model/lossof'..netname..'.t7',losses)

print('training end')
end
