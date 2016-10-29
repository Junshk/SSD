require 'cunn'
require 'cudnn'
require 'etc'
require 'image'
require 'pascal'
require 'prior_box'

torch.setdefaulttensortype('torch.FloatTensor')
--local real_box_redefine =real_box_ratio:t():view(1,4,20097):copy()
--local matio = require 'matio'
--local test_txt
--local test_list = {}
local softmax = cudnn.LogSoftMax():cuda()
softmax:evaluate()

--------------------------------------------
function write_txt(result,folder,class_num)
  assert(folder~=nil, 'need folder name to write txt file')
   -- write in filename txt 
      if paths.dirp(folder) ==false then os.execute('mkdir '..folder) end

--      for class_num = 1, 20 do

        local class_box , class_score = result.box, result.score
        if class_box:numel() ==0 then return end
        local write_result = io.open(folder..'/'..'comp3_det_test_'..num2class(class_num)..'.txt',"a")
    --      for image_iter = 1, #class_result do
          --   local image_result = class_result[image_iter]         
            local image_name = string.sub(result.image_name,-11,-1)

            for iter2 = 1, class_box:size(1) do
            --local box = result.box[{iter2}]
            local box = class_box[{{iter2}}]:squeeze()
            write_result:write(image_name,' ',result.score[{iter2}],' ',box[1],' ',box[2],' ',box[3],' ',box[4],'\n' )

            end
      --    end
        
--      end
  end

---------------------------------------------

function test(net,list,folder)
  local i1 = os.time()
  net:evaluate()

  print('testing..')
  local list = list or test_list
  local result = {}

  for iter = 1, 20 do table.insert(result,{}) end
  
  local result_vector = torch.Tensor(#list,25,20097)

  local batch = 12
  local iter = 1

-- forward
  while iter <= #list do
   
  local start_iter , end_iter = iter, math.min(iter +batch -1,#list)
  local n = end_iter - start_iter +1
  local input_tensor = torch.Tensor(n,3,500,500)
          -- input --
    for iter = start_iter ,end_iter do
    local imagename = list[iter].image_name

  
    local img = image.load(imagename..'.jpg')

    input_tensor[{{iter-start_iter+1}}] = image.scale(img,500,500)
  
    end
          -----------
          --forward--
  local output =net:forward(input_tensor:cuda()):float()
 -- result_vector[{{start_iter,end_iter}}] = output:float():squeeze()
  
  local conf_before_softmax = output[{{},{1,21}}]:transpose(2,3):reshape(n*20097,21)
  local conf = softmax:forward(conf_before_softmax:cuda()):view(n,20097,21):exp():float()
  
  local refined_box = output[{{},{22,25}}]

  refined_box = refined_box + real_box_ratio:view(1,4,20097):expand(n,4,20097)
  refined_box =refined_box:transpose(2,3)
  local score, recognition = torch.max(conf,3)
  -- nms
  for iter_image = 1, n do
    
    local image_name = list[iter_image+start_iter-1].image_name
    local size = image.load(image_name..'.jpg'):size()
    for iter_class =1, 20 do
    local res = {}
    local index = torch.eq(recognition[{iter_image,{},{}}],iter_class)
   -- print(index:size(),refined_box[iter_image]:size())
    local detection_box = refined_box[iter_image]
-- print(detection_box:size(),index:type())   
    detection_box =detection_box[index:expandAs(detection_box)]
 
    detection_box = detection_box:view(-1,4)
    local detection_score = score[iter_image][index]:view(-1)
    res.image_name = image_name
    
    --nms
    res.box, res.score = nms(detection_box,0.45,detection_score,size)

    write_txt(res, folder,iter_class)
    
    end

  end





  input_tensor =nil; collectgarbage();
  result = {}
  iter = end_iter + 1
  end
--[[
  net:float()

-- softmax
  iter = 1
  batch = 200
  local i2 =os.time()
  while iter <= #list do
  local s_iter, e_iter = iter, math.min(iter +batch-1,#list)
  local n = e_iter - s_iter +1
  local partial_vector = result_vector[{{s_iter,e_iter},{1,21},{}}]:transpose(2,3):squeeze()
  
  partial_vector = partial_vector:reshape(n*20097,21)
  partial_vector = softmax:forward(partial_vector:cuda()):view(n,20097,21):exp()

  result_vector[{{s_iter,e_iter},{1,21},{}}] = partial_vector:transpose(2,3):float()
  
  iter = e_iter + 1
  end


 local i3 = os.time()
 -- post processing
  print(' nms..')
  for iter = 1, #list do

  local image_size = image.load(list[iter].image_name..'.jpg'):size()

  if iter %1000 ==0 then  print(iter,#list) end
  
   local score, recognition = torch.max(result_vector[{{iter},{1,21}}]:squeeze(),1)
   local refined_box = real_box_ratio + result_vector[{{iter},{22,25}}]:squeeze()

    local detection_result ={}

    for class_num =1, 20 do
      local detection = {}
      local index = torch.eq(recognition,class_num)
   
      local class_detection_box = refined_box[index:expandAs(refined_box)]:view(-1,4)

      local class_score = score[index]
      local result_box,result_score = nms(class_detection_box,0.45,class_score,image_size)

      detection = {box = result_box, score = result_score,image_name = list[iter].image_name}
      table.insert(result[class_num],detection)
    end
    
   
  end

  local i4 =os.time()
  
  if folder ==nil then
  return result
  else -- write in filename txt 
      if paths.dirp(folder) ==false then os.execute('mkdir '..folder) end

      for class_num = 1, 20 do

        local class_result = result[class_num]
        local write_result = io.open(folder..'/'..'comp3_det_test_'..num2class(class_num)..'.txt',"a")
          for image_iter = 1, #class_result do
             local image_result = class_result[image_iter]         
            local image_name = string.sub(image_result.image_name,-11,-1)

            for iter2 = 1, image_result.box:size(1) do
            local box = image_result.box[{iter2}]
            write_result:write(image_name,' ',image_result.score[{iter2}],' ',box[1],' ',box[2],' ',box[3],' ',box[4],'\n' )

            end
          end
        
      end
  end
]]--
--  print('test time per a image : SSD',(i2-i1)/#list,'SOFTMAX',(i3-i2)/#list,'NMS',(i4-i3)/#list)
local i5 = os.time()
print('tot test time per an image',(i5-i1)/#list)
  net:training()
  net:cuda()


end

--map



function validation(net,savename)

local valid_txt ='VOCdevkit/VOC2012/ImageSets/Main/val.txt'
local valid_list = {}

local f = assert(io.open(valid_txt,'r'))

io.input(f)
--for i =1 ,5 do--
while  true do
local img ={}
local line = io.read()
if line ==nil then break end
img.image_name = 'VOCdevkit/VOC2012/JPEGImages/'..line
table.insert(valid_list,img)

end
io.close(f)


-- random sample list
local rand = torch.range(1,#valid_list)
local n = 500
local randperm = torch.randperm(n)
rand = rand:index(1,randperm:long())
local new_list ={}
for iter = 1, n do
new_list[iter] = valid_list[rand[iter]]
end

local result = test(net,new_list,'validation/'..savename)


end
