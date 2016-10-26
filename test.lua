require 'cunn'
require 'cudnn'
require 'etc'
require 'image'
require 'pascal'

local result_txt = 'result.txt' 
local test_list = ImgInfo('VOCdevkit/VOC2012_test/')

--local result_img_folder = 'result_img/'
--------------------------------------------




---------------------------------------------

function test(net,list)

local list = list or test_list


for iter = 1, #list do

local imagename = list[iter].image_name
print(imagename)

local img = image.load(imagename)
local img_size = img:size()
img_scaled = image:scale(img,500,500)

local result_vector = net:forward(img_scaled:cuda())



local result_box = nms(result_vector,0.5)

--reconstruction







end

end



function validation(net,savename)

local valid_txt = 'VOCdevkit/VOC2012/ImageSets/Main/val.txt'
local valid_list = {};

-- txt read


local reslut = test(net,valid_list)


torch.save(savename,result)


end
