require 'cunn'
require 'cudnn'

torch.setdefaulttensortype('torch.FloatTensor')

----------------------------------------
function bat2wo(input)
if input:dim() ==2 then return input end
local size = input:size()
local reshaped = input:transpose(2,3):reshape(1,size[1]*size[3],size[2]):squeeze(1)
--print(input:size())
--print(reshaped:size())
return reshaped
end
----------------------------------------
function wo2bat(input,batch)

local size = input:size()
assert(input:dim()==2,'wo2bat size error ')
local class = size[2]
local n = size[1]/batch

--print(input:size(),batch,n,class)

local reshaped = input:reshape(batch,n,class):transpose(2,3)
--print('re',reshaped:size())


return reshaped
end
-----------------------------------------
function MultiBoxLoss(input,target)  -- target1 : class 1 by pyramid, bd 4 by pyramid

local time0 =os.time()
local alpha = 1
--print(input:size())

local loss = 0
local batch ; if input:dim()==3 then batch= input:size(1) else batch=1 end


local element = torch.numel(target[1])
local negative_mask = torch.eq(target[1],21)
local positive_num = element-torch.sum(negative_mask)
local negative_num = math.min(element-positive_num,positive_num*3)
local discard_negative_num = (element)-positive_num-negative_num
local discard_mask, bd_conf

local input1 , input2 = bat2wo(input[{{},{1,21}}]),bat2wo(input[{{},{22,25}}])
local target1, target2 =bat2wo(target[1]), bat2wo(target[2])
negative_mask = bat2wo(negative_mask)

local softmax_result = torch.exp(nn.LogSoftMax():cuda():forward(input1):float())[{{},{21}}]

--print(discard_negative_num)

if discard_negative_num~=0 then
local score, k = torch.topk(softmax_result,discard_negative_num,1,true,true)
bd_conf = score[{{discard_negative_num}}]:squeeze()
else bd_conf =0 end
--print(bd_conf,discard_negative_num)
discard_mask = torch.gt(softmax_result,bd_conf):cmul(negative_mask)

--discard
input2[negative_mask:expandAs(input2)] =0
target2[negative_mask:expandAs(target2)]=0--//
--------------------------------
local dl_dx_loc = torch.Tensor(target2)
local dl_dx_conf 

local L1loss = nn.SmoothL1Criterion()
L1loss.sizeAverage = false
local CrossEntropy = nn.CrossEntropyCriterion()
local loss_conf ,loss_loc =0,0

L1loss:cuda(); --CrossEntropy:cuda() 
input:float()
-- forward



loss_conf = CrossEntropy:forward((input1):float(),(target1):float())
--[[
for i = 1, input2:size(1) do
loss_loc =loss_loc+ alpha*L1loss:forward((input2[{{i}}]):cuda(),(target2[{{i}}]):cuda())

end

for i = 1,input2:size(1) do
dl_dx_loc[{{i}}] = alpha * L1loss:backward((input2[{{i}}]):cuda(),(target2[{{i}}]):cuda()):float()
end]]--

loss_loc =alpha*L1loss:forward((input2):cuda(),(target2):cuda())/4
dl_dx_loc = alpha * L1loss:backward((input2):cuda(),(target2):cuda()):float()/4


input2:float() ; target2:float()

--print('for time', os.time()-time0)

dl_dx_conf = CrossEntropy:backward((input1):float(),(target1:float())):float()

input1:float(); target1:float()
--print(dl_dx_conf)
-- mining

--print(input[1]:size(),target[1]:size())

--discard conf
dl_dx_conf[discard_mask:expandAs(dl_dx_conf)] =0

--resize

dl_dx_loc = wo2bat(dl_dx_loc,batch)
dl_dx_conf = wo2bat(dl_dx_conf,batch)

--print(dl_dx_conf)
local dl_dx = torch.cat({dl_dx_conf,dl_dx_loc},2)
local n = positive_num+negative_num+1e-10 ; --if n ==0 then n =1; print('n ==0')end 



--print(loss_conf,loss_loc)
return (loss_conf+loss_loc)/n, dl_dx/n
end






