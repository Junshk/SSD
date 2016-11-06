
dofile('dataload.lua')

print('load datas')
var_w = 5
var_x = 10
norm = 1
Sub = true
logarithm = false
logadd = 1e-2
bgr = true
truck = true

logarithm = logarithm and not Sub
print(var_w,var_x,norm,logarithm,Sub,bgr,truck)
-----------------------

require 'training'
local i = 1000

local option =
{
  
  netname = 'vgg_SSD500_varIsee_v2_prior_reAug_cropsxy',

  plot_iter =100,end_iter = 80*1000,
  print_iter=1,save_iter=200,
  test_iter = i,
  batch_size = 12, multi_batch =2,
  valid =true,
  cont = true
, ch = true
, mul = true
, lambda =1
}

--option.netname = option.netname..'_b'.. option.batch_size..'_m'..option.multi_batch.. '_lam'..option.lambda*100
--if option.ch == true then option.netname  = option.netname .. '_ch' end

--if option.mul == true then option.netname  = option.netname .. '_mul_fixconf' end
-- training
if Sub == true then option.netname = option.netname .. '_sub' end
if truck == true then option.netname = option.netname .. '_truck' end
if logarithm == true then option.netname = option.netname .. 'log' end
option.netname = option.netname ..'_w'..var_w..'_x'..var_x


training(option)

-- test code
require 'test'
local test_txt = 'VOCdevkit/VOC2012_test/ImageSets/Main/test.txt'
local test_list = {}
local f_test = assert(io.open(test_txt,"r"))

for line in io.lines(f_test) do
        table.insert(test_list,line)
end
f_test:close()

local test_folder = 'Test/'
local net = torch.load(option.netname..'.net')

test(net,test_list,test_folder)

os.execute('matlab -r -nodisplay plot_map')
