require 'optim'
--require 'make_net'
require 'option'
require 'gnuplot'
require 'FetchData'
require 'MultiBoxLoss'
require 'test'
require 'os'


--require 'prior_box'
--SGD
optimState ={

learningRate = 1e-3,
momentum = 0.9,
weightDecay = 0.0005

}
local iteration 
local batch_num 
local data_num = data_num--5e4
--local extract_idx = torch.randperm(data_num)
local opt = Option
local batch_size = opt.batch_size
local multi_batch = opt.multi_batch 


local inputs = torch.Tensor(batch_size,3,500,500)
local targets = torch.Tensor(batch_size,5,20097)
local batch_idx = 1
torch.manualSeed(os.time())
local rand = torch.randperm(data_num)-1

function training()
--math.randomseed(os.time())
----------------------------
print(opt)
print(net)
print('training')

local params, grads = net:getParameters()

local feval = function(x)

if x ~= params then
        params:copy(x)
        end

local f =0
local acc =0
local boxN =0
local PN =0
local acc_n =0



grads:zero()


assert(torch.sum(torch.ne(inputs,inputs))==0 , 'nan in input')

input_af = pretrain:forward(inputs:cuda())
local output = net:forward(input_af:cuda())
inputs = inputs:float()

-----------------------------------
assert(torch.sum(torch.ne(output,output))==0, 'nan in output')

local err, df_dx,N, accuracy,accuracy_n = MultiBoxLoss(output,targets,opt.lambda)
output = output:float()


f = f+ err
boxN = boxN +N
acc = acc +accuracy
acc_n = acc_n +accuracy_n

net:backward(input_af:cuda(),df_dx:cuda())

--end -- for end
input_af =nil
--targets =nil
output = nil
collectgarbage()
if boxN ==0 then 
f = 0
table.insert(accuracies,accuracies[#accuracies])
return f, grads:fill(0)
else
table.insert(accuracies,acc/acc_n)
f = f/ boxN
grads:div(boxN)
return f, grads
end

collectgarbage();
end -- end local feval


for iteration = start_iter, opt.end_iter do

local rand_st = batch_size * (batch_idx - 1)

if rand_st +batch_size > data_num then
  batch_idx = 1
  rand_st = batch_size *(batch_idx - 1)
  torch.manualseed(os.time())
  rand = torch.randperm(data_num) - 1
end

for donkeyAdd = 1, batch_size do
donkeys:addjob(
                 function()
                
                 local idx = rand[rand_st + donkeyAdd]--torch.randperm(data_num)
                 local name_i = 0
                 ::resample:: 
               --  print(idx,donkeyAdd)
                 
             
                  
                 if paths.filep('data/SSDdata_'..idx..'.t7') == false then 
                   --math.randomseed(os.time());
                   idx = math.random(data_num) 
                   goto resample
                 end
                 local data_name = 'data/SSDdata_'..idx..'.t7'
                 

                 local data = torch.load(data_name)
                 
                 
                 input = data.input
                 target = data.target
                 
                 
                 collectgarbage()
                 return input, target, donkeyAdd
                 end
                , 
                 trainBatch
                 
                 )
 end
  donkeys:synchronize() 
 batch_idx = batch_idx + 1
 
 local _, loss = optim.sgd(feval,params,optimState)
  
 


  if iteration == 60*1000 then optimState.learningRate = 1e-4 end
     if iteration % opt.print_iter ==0 then 
        print('iter',iteration,'loss ',loss[1],'acc',accuracies[#accuracies])

  end

   if opt.valid ==true and iteration%opt.test_iter ==0 then
        local folder_ = 'validation/'..netname..'/valid_loss_'..iteration
         if paths.dirp(folder_) ==true then  os.execute('rm -r '..folder_) end
         validation(net,'valid_loss_'..iteration,netname)   
  end
 
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
------------------------------------
function trainBatch(inputCPU,targetCPU,donkeyAdd)

collectgarbage()


inputs[{{donkeyAdd}}] = inputCPU
targets[{{donkeyAdd}}] = targetCPU

batch_idx = batch_idx + 1
if batch_idx > batch_size  then batch_idx = 1 end

end
------------------------------------------
-------------------------------------------



