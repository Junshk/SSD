require 'nn'






local SSDloss,parent = torch.class('nn.SSDloss','nn.Modules')

-- batch by default by class or 4

function SSDloss:__init(default)

assert(type(default)=='number','class must be a number')
self.cuda =false
self.default = default
self.softmax = nn.CrossEntropyCriterion()
self.l1 = nn.SmoothL1Criterion()
end
--[[
function SSDloss:cuda()
self.softmax:cuda()
self.l1:cuda()
self.cuda =true
end
]]--
function SSDloss:updateOutput(input,target) -- x,c,/l,g  --4d

assert(type(input) == 'table' and  type(target)=='table','loss input,target type == table')

local c = input[1] -- batch by default by class
local l = input[2] -- t_i+delta

local x = target[1] -- 1~21 batch by default
local g = target[2]

local batch = c:size(1)
assert(x:dim()==3 and l:dim()==3 and c:dim()==3 and g:dim()==3,'3d data')
assert(x:size(2)==self.default and l:size(2)==self.default and 
c:size(2)==self.default and g:size(2) ==self.default , 'inconsistent with default size')


local negative_pos = torch.eq(x[{{},{},{21}}],1)
local positive = torch.sum(torch.sum(x[{{},{},{1,20}}],3),2):squeeze() --batch
local nMax_idx = torch.gt(positive*3,self.default-positive)
local negativeMax = positive*3 ; negativeMax[nMax_idx] = self.default-positive[nMax_idx]

local _,sort_idx = torch.max(c[{{},{},{21}}],2)
local topk_idx = sort_idx:index(2,negativeMax).....

local matchingBox = x[]


self.N = torch.sum(x,2); -- default by 1


local loss_softmax = torch.Tensor(self.default):fill(0)
local loss_l1 = torch.Tensor(self.default):fill(0)

for it_bt = 1, batch do

for iter = 1, self.default do
if self.cuda ==false then
loss_softmax[{{iter},{it_bt}}] = self.softmax:forward(c[{{iter,{it_bt}}}],x[{{iter,{it_bt}}}]) --output,target
loss_l1[{{iter}}] = self.l1:forward(l[{{iter}}],g[{{iter}}])

--[[
elseif self.cuda == true then 
loss_softmax[{{iter}}] = self.softmax:forward(c[{{iter}}]:cuda(),x[{{iter}}]:cuda()) --output,target
loss_l1[{{iter}}] = self.l1:forward(l[{{iter}}]:cuda(),g[{{iter}}]:cuda())
]]--
end
end

end -- batch for end

return (loss_softmax+loss_l1):float():cdiv(self.N)

end

function SSDloss:updateGradInput(input,target) -- x,c,/l,g  --4d

assert(type(input) == 'table' and  type(target)=='table','loss input,target type == table')

local c = input[1] -- default by class
local l = input[2] -- t_i+delta

local x = target[1] 
local g = target[2]

assert(x:dim()==3 and l:dim()==3 and c:dim()==3 and g:dim()==3,'3d data')
assert(x:size(2)==self.default and l:size(2)==self.default and 
c:size(2)==self.default and g:size(2) ==self.default , 'inconsistent with default size')


--self.N = torch.sum(x,2); -- default by 1


local grad_softmax = torch.Tensor(c:size()):fill(0)
local grad_l1 = torch.Tensor(l:size()):fill(0)

for iter = 1, self.default do

if self.cuda ==true then
grad_softmax[{{iter}}] = self.softmax:backward(c[{{iter}}]:cuda(),x[{{iter}}]:cuda()) --output,target
grad_l1[{{iter}}] = self.l1:backward(l[{{iter}}]:cuda(),g[{{iter}}]:cuda())
elseif self.cuda ==false then
grad_softmax[{{iter}}] = self.softmax:backward(c[{{iter}}],x[{{iter}}]) --output,target
grad_l1[{{iter}}] = self.l1:backward(l[{{iter}}],g[{{iter}}])

end
end
-- mining 
return grad_softmax/self.N,grad_l1/self.N
end


