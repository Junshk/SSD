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

function wo2bat(input,batch)
--[[
local size = input:size()

local class = size[2]
local n = size[1]/batch
print(input:size(),'i')
local reshaped = input:reshape(batch,n,class):transpose(2,3)

print(input:size(),reshaped:size(),batch,n,class)
]]--

return input
end
-----------------------------------------
function MultiBoxLoss(input,target)  -- target1 : class 1 by pyramid, bd 4 by pyramid

local alpha = 1
--print(input,target)

local loss = 0
local dl_dx_loc = {}
local dl_dx_conf = {}
local batch ; if input[1]:dim()==3 then batch= input[1]:size(1) else batch=1 end

local L1loss = nn.SmoothL1Criterion()
L1loss.sizeAverate = false
local CrossEntropy = nn.CrossEntropyCriterion()
local loss_conf ,loss_loc =0,0

L1loss:cuda(); CrossEntropy:cuda() 
input[1]:float(); input[2]:float()

print('loss init')
-- forward

print('loss loc')

loss_loc= alpha*L1loss:forward(bat2wo(input[2]):cuda(),bat2wo(target[2]):cuda())
dl_dx_loc = alpha * L1loss:backward(bat2wo(input[2]):cuda(),bat2wo(target[2]):cuda()):float()
input[2]:float() ; target[2]:float()
print('loss conf')

loss_conf = CrossEntropy:forward(bat2wo(input[1]):cuda(),bat2wo(target[1]):cuda())
print(1)
dl_dx_conf =  CrossEntropy:backward(bat2wo(input[1]):cuda(),bat2wo(target[1]):cuda()):float()
input[1]:float(); target[1]:float()
--print(dl_dx_conf)
-- mining


--collectgarbage();



return loss_conf+loss_loc, dl_dx_conf, dl_dx_loc

end






