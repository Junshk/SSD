require 'nn'

----------------------------------------


function MultiBoxLoss(input,target)  -- target1 : class 1 by pyramid, bd 4 by pyramid

local alpha = 1

local pyramid_level = #target[2]
local loss = torch.Tensor(pyramid_level)
local dl_dx_loc = {}
local dl_dx_conf = {}

local L1loss = nn.SmoothL1Criterion()
L1loss.sizeAverate = false
local CrossEntropy = nn.CrossEntropyCriterion()

--local matchingDefaultNum = torch.numel()

if torch.type(input) == 'torch.CudaTensor' then L1loss:cuda(); CrossEntropy:cuda() end


for iter = 1,pyramid_level do 

local element = torch.numel(target[1][iter])
if torch.type(input) == 'torch.CudaTensor' then target[1][iter]:cuda(); target[2][iter]:cuda() end

-- forward
loss[iter]= alpha * L1loss:forward(input[2][iter]:reshape(4,element):t(),target[2][iter]:reshape(4,element):t())+
CrossEntropy:forward(input[1][iter]:reshape(21,element):t(),target[1][iter]:reshape(element))

-- backward
dl_dx_loc[iter] = alpha * L1loss:backward(input[2][iter]:reshape(4,element):t(),target[2][iter]:reshape(4,element):t())

dl_dx_conf[iter] =  CrossEntropy:backward(input[1][iter]:reshape(21,element):t(),target[1][iter]:reshape(element))


-- mining


collectgarbage();
end


return loss:sum(), {dl_dx_conf, dl_dx_loc}

end






