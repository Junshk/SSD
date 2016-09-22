require 'optim'
require 'make_net'


--SGD
optimState ={

learningRate = 1e-3,
momentum = 0.9,
weightDecay = 0.0005

}

local batch_size = 32


function training(basenet)
local net = make_net(basenet)





net:clearState()
torch.save('model/'..basenet..'SSDnet.t7',net)
end
