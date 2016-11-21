
dofile('dataload.lua')

print('load datas')
var_w = 1--5
var_x = 1--10
norm = 1
logarithm = true
bgr = true
truck = true
chm = true--false

r_mean = 123.68
g_mean = 116.779
b_mean = 103.939

weighted21 =false
random_data2 = true
logarithm = logarithm and not Sub
print(var_w,var_x,norm,logarithm,Sub,bgr,truck)
-----------------------

require 'training'
local i =  500

local option =
{
  
  netname = 'vgg_SSD500_1121'  --depart pretrain fixed w 
,  plot_iter =50,end_iter = 80*1000,
  print_iter=1,save_iter=100,
  test_iter = i,
  batch_size = 12, multi_batch =2,
  valid =true,
  cont = true--false
, ch = true
, mul = true
, lambda =1
}

-- training
if weighted21 == true then option.netname = option.netname .. '_weg21' end
--if chm == false then option.netname = option.netname .. '_nochm'end
--if logarithm == true then option.netname = option.netname .. 'log' end
option.netname = option.netname ..'_w'..var_w..'_x'..var_x..'_n'..norm
if random_data2 == true then option.netname = option.netname..'_rdat2' end
if paths.filep('VGG16.net') ==false then 
  dofile('utils/caffe.lua')
end
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
