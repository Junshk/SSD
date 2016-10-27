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
local softmax = cudnn.LogSoftMax():cuda()


--------------------------------------------
---------------------------------------------

function test(net,list,folder)
local i1 = os.time()
net:evaluate()

  print('testing..')
  local list = list or test_list
  local result = {}

  for iter = 1, 20 do table.insert(result,{}) end
  
  local result_vector = torch.Tensor(#list,25,20097)

  local batch = 14
  local iter = 1

-- forward
  while iter <= #list do
   
  local start_iter , end_iter = iter, math.min(iter +batch -1,#list)
  local input_tensor = torch.Tensor(end_iter-start_iter+1,3,500,500)
  
    for iter = start_iter ,end_iter do
    local imagename = list[iter].image_name

  
    local img = image.load(imagename..'.jpg')

    input_tensor[{{iter-start_iter+1}}] = image.scale(img,500,500)
  
    end
  
  result_vector[{{start_iter,end_iter}}] = net:forward(input_tensor:cuda()):float():squeeze()

  input_tensor =nil; collectgarbage();
  
  iter = end_iter + 1
  end

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
  net:training()
  net:cuda()

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

print('test time : ',i2-i1,i3-i2,i4-i3)

end

--map



function validation(net,savename)

local valid_txt ='VOCdevkit/VOC2012/ImageSets/Main/val.txt'
local valid_list = {}

local f = assert(io.open(valid_txt,'r'))

io.input(f)
--for i =1 ,50 do--
while  true do
local img ={}
local line = io.read()
if line ==nil then break end
img.image_name = 'VOCdevkit/VOC2012/JPEGImages/'..line
table.insert(valid_list,img)

end
io.close(f)

local result = test(net,valid_list,'validation/'..savename)


end
