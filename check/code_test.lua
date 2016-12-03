require 'make_net'
require 'test'
require 'option'


local netname = option.netname..'_intm'


local net = torch.load('model/'..netname..'.net')
local savename = 'code_test/'
validation(net,savename,netname)


