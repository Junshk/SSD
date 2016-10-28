
dofile('dataload.lua')

require 'training'

print('load datas')

local i = 400

local option =
{
  
  netname = 'vgg_SSD500',

  plot_iter =100,end_iter = 80*1000,print_iter=10,save_iter=80,
  test_iter = i,
  batch_size = 12, multi_batch =3,
  valid =true,
  cont =true
, ch = false
}

option.netname = option.netname..'_b'.. option.batch_size..'_m'..option.multi_batch
if option.ch == true then option.netname  = option.netname .. '_ch' end
-- training
training(option)

-- test code
require 'test'
local test_list = {}
local test_folder = 'Test/'
local net = torch.load(option.netname..'.net')

test(net,test_list,test_folder)

os.excute()
