require 'cunn'
require 'cudnn'
require 'etc'
require 'image'
require 'pascal'
require 'prior_box'

torch.setdefaulttensortype('torch.FloatTensor')

local matio = require 'matio'
local test_txt
local test_list = {}


--------------------------------------------
---------------------------------------------

function test(net,list)
  net:evaluate()

  print('testing..')
  local list = list or test_list
  local result = {}

  local result_vector = torch.Tensor(#list,25,20097)

  local batch = 14
  local iter = 1

-- forward
  while iter <= #list do
   
  local start_iter , end_iter = iter, math.min(iter +batch -1,#list)
  local input_tensor = torch.Tensor(end_iter-start_iter+1,3,500,500)
  
    for iter = start_iter ,end_iter do
    local imagename = list[iter].image_name
--    print(imagename)
  
    local img = image.load(imagename..'.jpg')

    input_tensor[{{iter-start_iter+1}}] = image.scale(img,500,500)--view(1,3,500,500)
  
    end

  result_vector[{{start_iter,end_iter}}] = net:forward(input_tensor:cuda()):float():squeeze()
  input_tensor =nil; collectgarbage();
  
  iter = end_iter + 1
  end

 -- post processing

  for iter = 1, #list do
  if iter %1000 ==0 then  print(iter,#list) end
   local score, recognition = torch.max(result_vector[{{iter},{1,21}}]:squeeze(),1)
   local refined_box = real_box_ratio + result_vector[{{iter},{22,25}}]:squeeze()

    local detection_result ={}

    for class_num =1, 20 do
      local index = torch.eq(recognition,class_num)
   
      local class_detection_box = refined_box[index:expandAs(refined_box)]:view(-1,4)

      local class_score = score[index]
      local result_box = nms(class_detection_box,0.45,class_score)
      detection_result[num2class(class_num)] = {box = result_box, score = class_score}
    end
    
    local result_per_image = {detection = detection_result, image_name = list[iter].image_name}
--  print(result_per_image)
  table.insert(result, result_per_image)
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

for i =1 ,1 do--while  true do
local img ={}
local line = io.read()
if line ==nil then break end
img.image_name = 'VOCdevkit/VOC2012/JPEGImages/'..line
table.insert(valid_list,img)

end

io.close(f)

local result = test(net,valid_list)
print(savename,result)
matio.save(savename..'.mat',result)

end
