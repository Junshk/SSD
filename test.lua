require 'cunn'
require 'cudnn'
require 'etc'
require 'image'
require 'pascal'

local result_txt = 'result.txt' 
local list = ImgInfo('VOCdevkit/VOC2012_test/')
local net = torch.load('model/vggSSDnet_intm.t7')
local result_img_folder = 'result_img/'
--------------------------------------------
function write_box()
end
function drawing_box()
end
---------------------------------------------
for iter = 1, #list do

local imagename = list[iter].image_name
print(imagename)

local img = image.load(imagename)
local img_size = img:size()
img_scaled = image:scale(img,500,500)

local result_vector = net:forward(img_scaled:cuda())



local result_box = nms(result_vector,0.5)

--reconstruction
result_box = reconstruction(result_box,img_size())

local result_img = drawing_box(result_box,img)
image.save(result_img_folder..imagename..'.jpg',result_img)

write_box(result_box,result_txt)

end


