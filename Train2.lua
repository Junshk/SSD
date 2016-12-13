require 'optim'
--require 'make_net'
require 'option'
require 'gnuplot'
require 'FetchData'
require 'MultiBoxLoss'
require 'Test'
require 'os'


--require 'prior_box'
--SGD
optimState ={

learningRate = 1e-3,
momentum = 0.9,
weightDecay = 0.0005,
learningRateDecay =0,
dampening =0 ,
nesterov = true

}
local iteration 
local batch_num 
local data_num = data_num--5e4
--local extract_idx = torch.randperm(data_num)
local opt = Option
local batch_size = opt.batch_size
local multi_batch = opt.multi_batch 


--local inputs = torch.Tensor(batch_size,3,500,500)
--local targets = torch.Tensor(batch_size,5,20097)
--local batch_idx = 1
local f =0
local acc =0
local boxN =0
local PN =0
local acc_n =0




function training()
math.randomseed(os.time())
----------------------------
print(opt)
print(pretrain)
print(net)
print('training')

local params, grads = net:getParameters()

local feval = function(x)

if x ~= params then
        params:copy(x)
        end

--print('g',boxN,f,acc,acc_n)
if boxN ==0 then 
f = 0
table.insert(accuracies,accuracies[#accuracies] or 0)

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

  f =0
 acc =0
 boxN =0
 PN =0
 acc_n =0



grads:zero()
--print(grads[])
--print(f)
--[[
for donkeyAdd = 1, multi_batch do
donkeys:addjob(
                 function()
                 torch.manualSeed(os.time())
                 --math.randomseed(os.time())
                 --local inputs, targets = torch.Tensor(batch_size,3,500,500), torch.Tensor(batch_size,5,20097)
                 ::resample::
                 local rand = torch.randperm(data_num+1)-1 
                 
                 local idx = rand[donkeyAdd]--torch.randperm(data_num)
                 local name_i = 0
                -- print(idx,donkeyAdd)
                 
             
                  
                 --if paths.filep('data/SSDdata_'..idx..'.t7') == false then goto resample end
                 --local data_name = 'data/SSDdata_'..idx..'.t7'
                 

                 --local data = torch.load(data_name)
                 
                 
                 local input, target =  patchFetch(batch_size,img_Info_table,iteration*donkeyAdd)
                 
                 
                 
                 collectgarbage()
                 return input, target, donkeyAdd
                 end
                , 
                 trainOne
                 
                 )
 end
donkeys:synchronize()

]]--
  for batch_iter = 1, multi_batch do
   local input_, target_ = patchFetch(batch_size,img_Info_table,iteration*multi_batch+batch_iter)
   trainOne(input_,target_)
  end
  
 local _, loss = optim.sgd(feval,params,optimState)
  


  if iteration == 60*1000 then optimState.learningRate = 1e-4 end
     if iteration % opt.print_iter ==0 then 
        print('------------------------------------------------------------')
        print('iter',iteration,'loss ',loss[1],'acc',accuracies[#accuracies],netname)
        print('------------------------------------------------------------')
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
function trainOne(input,target,donkeyAdd)

collectgarbage()
--[[

inputs[{{donkeyAdd}}] = inputCPU
targets[{{donkeyAdd}}] = targetCPU
batch_idx = batch_idx + 1
if batch_idx > batch_size  then batch_idx = 1 end
]]--

--print(input:size())
assert(torch.sum(torch.ne(input,input))==0 , 'nan in input')

local input_af = pretrain:forward(input:cuda())
local output = net:forward(input_af:cuda())
--input = nil--input:float()

-----------------------------------
--assert(torch.sum(torch.ne(output,output))==0, 'nan in output')

local err, df_dx,N, accuracy,accuracy_n = MultiBoxLoss(output,target,opt.lambda)
--output = nil--output:float()


f = f+ err
boxN = boxN +N
acc = acc +accuracy
acc_n = acc_n +accuracy_n

net:backward(input_af:cuda(),df_dx)

--end -- for end
--input_af =nil
--targets =nil
--output = nil
collectgarbage()




end
------------------------------------------
-------------------------------------------



