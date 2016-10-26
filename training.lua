require 'optim'
require 'make_net'

require 'gnuplot'
require 'FetchData'
require 'MultiBoxLoss'
require 'test'
print('training')
cutorch.setDevice(1)

--require 'prior_box'
--SGD
optimState ={

learningRate = 1e-3,
momentum = 0.9,
weightDecay = 0.0005

}

local batch_size = 15 
-------------------------------------------------------------------------------

function training(opt)

local basenet = 'vgg'
if paths.dirp('model') ==false then os.execute('mkdir model') end
local net = make_net(basenet)
local netname = basenet .. '_b'.. batch_size
net:training()
net:cuda()
cudnn.convert(net,cudnn)



local img_Info_table = ImgInfo()


local params, grads = net:getParameters()

local feval = function(x)

if x ~= params then
        params:copy(x)
        end



local input, target = patchFetch(batch_size,img_Info_table) --imgtensor, table

grads:zero()

local detc = torch.sum(torch.gt(target[1],21))+ torch.sum(torch.lt(target[1],1))
assert(detc==0 , 'wrong class label')

local output = net:forward(input:cuda()):float()
input:float()



-----------------------------------

local err, df_dx = MultiBoxLoss(output,target)
------------------------------------

net:backward(input:cuda(),df_dx:cuda())


collectgarbage();
return err,grads
end -- end local feval
-------------------------------------------

local losses = {}
local val_losses = {}
for iteration =1,opt.end_iter do


if iteration == 40*1000 then optimState.learningRate = 1e-4 end


local _, loss = optim.sgd(feval,params,optimState)

table.insert(losses,loss[1])
if opt.valid ==true and iteration%opt.test_iter ==0 then
         validation(net,'valid_loss_'..iteration)
        
    end
  if iteration%opt.plot_iter ==0 then
        local start_num, end_num = math.max(1,iteration-opt.plot_iter*10),iteration
        gnuplot.plot({'loss',torch.range(start_num,end_num),torch.Tensor(losses)[{{start_num,end_num}}],'-'})
  end

  if iteration % opt.print_iter ==0 then 
        print('iter',iteration,'loss ',loss[1])
  end


  if iteration % opt.save_iter ==0 then 
        net:clearState()
        torch.save('model/'..netname..'_intm.net',net)
        torch.save('loss/lossof'..netname..'_intm.t7',losses)
  end

end




  net:clearState()


  torch.save('model/'..netname..'.nnet',net)
  torch.save('loss/lossof'..netname..'.t7',losses)

print('training end')
end
