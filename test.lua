require 'cunn'
require 'cudnn'
require 'etc'
require 'image'
require 'pascal'
require 'prior_box'
local matio = require 'matio'
local test_txt
local test_list = {}


--------------------------------------------
---------------------------------------------



function test(net,list)
local list = list or test_list

net:evaluate()
local list = list or test_list
local result = {}

  for iter = 1, #list do

  local imagename = list[iter].image_name
  print(imagename)

  local img = image.load(imagename..'.jpg')
  local img_size = img:size()
  img_scaled = image:scale(img,500,500)

  local result_vector = net:forward(img_scaled:cuda())

  local score, recognition = torch.max(result_vector[{{},{1,21}}],2)

  local refined_box = real_box_ratio:t() + result_vector[{{},{22,25}}]

  local detection_result ={}

    for class_num =1, 20 do
    local index = torch.eq(recognition,class_num)
    local class_detection_box = refined_box[index]
    local class_score = score[index]
    local result_box = nms(class_detection_box,0.5,class_score)
    detection_result[class_num] = result_box
    end

    table.insert(result,detection_result)
  end


net:training()
return result

end

--map



function validation(net,savename)

local valid_txt ='VOCdevkit/VOC2012/ImageSets/Main/val.txt'
local valid_list = {}

local f = assert(io.open(valid_txt,'r'))
--local t = f:read()
io.input(f)

while  true do
local img ={}
local line = io.read()
if line ==nil then break end
img.image_name = line
table.insert(valid_list,img)

end

io.close(f)

local result = test(net,valid_list)

matio.save(savename..'.mat',result)

end
