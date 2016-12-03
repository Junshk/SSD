require 'cutorch'
cutorch.setDevice(1)
torch.setnumthreads(1)
------------------------
image_size = 500
fmap_n = 7
s_max = 0.9
s_min = 0.1

-------------------------
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

data_num = 9e4

print(var_w,var_x,norm,logarithm,bgr,truck)
-----------------------

local i =1000 
 Option =
{
  
  netname = 'vgg_SSD500_1130'
,  plot_iter =50,end_iter = 100*1000,
  print_iter=1,save_iter=50,
  test_iter = i,
  batch_size = 12, multi_batch =2,
  valid =true,
  cont =true-- false
, ch = true
, mul = true
, lambda =1
}

-- training
if weighted21 == true then Option.netname = Option.netname .. '_weg21' end
--if chm == false then option.netname = option.netname .. '_nochm'end
--if logarithm == true then option.netname = option.netname .. 'log' end
Option.netname = Option.netname ..'_w'..var_w..'_x'..var_x..'_n'..norm
if truck == false then Option.netname = Option.netname ..'_noTruck' end
--if random_data2 == true then Option.netname = Option.netname..'_rdat2' end
if paths.filep('VGG16.net') ==false then 
  dofile('utils/caffe.lua')
end

