require 'optim'
require 'make_net'

require 'gnuplot'
require 'FetchData'
require 'MultiBoxLoss'
require 'test'
require 'os'
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
local accuracies = {}
local start_iter = 1

if paths.filep('model/'..netname..'_intm.net') == false then 
opt.cont =false end
  
  local net, pretrain =  make_net(opt.ch,opt.mul)
  pretrain = cudnn.convert(pretrain,cudnn):cuda()
  pretrain:evaluate()

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

print(opt)
net:training()
--net:cuda()
print(net)
print('training')

local img_Info_table = ImgTxt('VOCdevkit/VOC2012','trainval.txt')--ImgInfo()--trainInfo()--ImgInfo()
img_Info_table = ImgTxt('VOCdevkit/VOC2007','trainval.txt',img_Info_table)
img_Info_table = ImgTxt('VOCdevkit/VOC2007','test.txt',img_Info_table)

print('list num:',#img_Info_table)

local params, grads = net:getParameters()

local feval = function(x)
::retrain::
if x ~= params then
        params:copy(x)
        end



grads:zero()
--net:clearState()
--local detc = torch.sum(torch.gt(target[1],21))+ torch.sum(torch.lt(target[1],1))
--assert(detc==0 , 'wrong class label')
local f =0
local acc =0
local boxN =0
local PN =0
local acc_n =0


for iter = 1, multi_batch do
local o1=os.time()
local input, target = patchFetch(batch_size,img_Info_table) --imgtensor, table
assert(torch.sum(torch.ne(input,input))==0 , 'nan in input')
local o2 =os.time()

input = pretrain:forward(input:cuda())

local output = net:forward(input:cuda())
input:float()

-----------------------------------
assert(torch.sum(torch.ne(output,output))==0, 'nan in output')

local err, df_dx,N, accuracy,accuracy_n = MultiBoxLoss(output:float(),target,opt.lambda)
print('o',o2-o1,os.time()-o2)
f = f+ err
boxN = boxN +N
acc = acc +accuracy
acc_n = acc_n +accuracy_n
------------------------------------
net:backward(input:cuda(),df_dx:cuda())


end
if boxN ==0 then goto retrain end
table.insert(accuracies,acc/math.max(acc_n,1))--boxN)
f = f/ boxN
grads:div(boxN)

collectgarbage();
return f, grads
end -- end local feval
-------------------------------------------
for iteration = start_iter,opt.end_iter do
 local t1 =os.time() 
  if iteration == 60*1000 then optimState.learningRate = 1e-4 end
  local _, loss = optim.sgd(feval,params,optimState)
    if iteration % opt.print_iter ==0 then 
        print('iter',iteration,'loss ',loss[1],'acc',accuracies[#accuracies])

  end

  local t2 =os.time()
   if opt.valid ==true and iteration%opt.test_iter ==0 then
        local folder_ = 'validation/'..netname..'/valid_loss_'..iteration
         if paths.dirp(folder_) ==true then  os.execute('rm -r '..folder_) end
         validation(net,'valid_loss_'..iteration,netname)   
  end
  print('t',t2-t1,os.time()-t2)
  losses[iteration] = loss[1]

 if iteration % opt.save_iter ==0 then 
        net:clearState()
        print('net saving')
        torch.save('model/'..netname..'_intm.net',net)
        torch.save('model/accof'..netname..'_intm.t7',accuracies)
        torch.save('model/lossof'..netname..'_intm.t7',losses)
        torch.save('model/optof'..netname..'.t7',opt)
        print('net saved')
   end


  if iteration%opt.plot_iter ==0 then
    local start_num, end_num = 
    math.max(1,iteration-opt.plot_iter*100), iteration
    gnuplot.figure(1)
    gnuplot.plot({netname..'loss',torch.range(start_num,end_num),torch.Tensor(losses)[{{start_num,end_num}}],'-'})
    
    gnuplot.figure(2)
    gnuplot.plot({netname..'acc',torch.range(1,#accuracies),torch.Tensor(accuracies),'-'})
  end

  
end




  net:clearState()

  torch.save('model/accof'..netname..'.t7',accuracies)
  torch.save('model/'..netname..'.net',net)
  torch.save('model/lossof'..netname..'.t7',losses)

print('training end')
end
