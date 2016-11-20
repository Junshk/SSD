require 'cunn'
require 'cudnn'
--require 'etc'
require 'image'
require 'pascal'
require 'prior_box'
print('test load')
local batch = 8

torch.setdefaulttensortype('torch.FloatTensor')

local softmax = cudnn.LogSoftMax():cuda()
softmax:evaluate()

local img_save_iter = 1
--------------------------------------------
function write_txt(result,folder,class_num)
  assert(folder~=nil, 'need folder name to write txt file')
   -- write in filename txt 
      if paths.dirp(folder) ==false then os.execute('mkdir '..folder) end

--      for class_num = 1, 20 do
        local bb_image = image.load(result.image_name.. '.jpg')
        local class_box , class_score = result.box, result.score
        if class_box:numel() ==0 then return end
        local write_result = io.open(folder..'/'..'comp3_det_test_'..num2class(class_num)..'.txt',"a")
        write_result:write('\n')
    --      for image_iter = 1, #class_result do
          --   local image_result = class_result[image_iter]         
            local image_name = string.sub(result.image_name,-11,-1)

            for iter2 = 1, class_box:size(1) do
            --local box = result.box[{iter2}]
            local box = class_box[{{iter2}}]:squeeze()
            write_result:write(image_name,' ',result.score[{iter2}],' ',box[1],' ',box[2],' ',box[3],' ',box[4],'\n' )
            bb_image = image.drawRect(bb_image,box[1],box[2],box[3],box[4])
            end
      --    end
       write_result:close() 
--      end
  
  image.save('conf/'..img_save_iter..'.jpg',bb_image)
  img_save_iter = img_save_iter+1
  end

---------------------------------------------

function test(net,list,folder)
  img_save_iter =1
  
  local i1 = os.time()
  net:evaluate()
  
  print('testing..')
  if paths.dirp(folder) ==false then os.execute('mkdir '..folder) end
 
  local newf = assert(io.open(folder..'/test.txt',"w"))
  for iter = 1, #list do
        newf:write(string.sub(list[iter].image_name,-11,-1),'\n')
  end
  newf:close()

  local list = list or test_list
  local result = {}

  for iter = 1, 20 do table.insert(result,{}) end
  
  local result_vector = torch.Tensor(#list,25,20097)

  local iter = 1
  --net:float()
-- forward
  while iter <= #list do
    print('test ',iter)
   if iter %1000 ==0 then  print(iter,'/',#list) end
 --print(start_iter) 
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
  --net:cuda()
  local output =net:forward(input_tensor:cuda()):float()
  --input_tensor=input_tensor:float()
  --input_tensor = nil; collectgarbage()
  --net:float()
 -- result_vector[{{start_iter,end_iter}}] = output:float():squeeze()
  
  local conf_before_softmax = output[{{},{1,21}}]:transpose(2,3):reshape(n*20097,21)
  local conf = softmax:forward(conf_before_softmax:cuda()):view(n,20097,21):exp():float()
  
  local refined_box = output[{{},{22,25}}]
  -- jihong --
  refined_box[{{},{1,2}}]:div(var_w)
  refined_box[{{},{3,4}}]:div(var_x)

  local expand = real_box_ratio:view(1,4,20097):expand(n,4,20097)

  if logarithm == true then refined_box[{{},{1,2}}]:exp() end
  refined_box[{{},{3,4}}]:cmul(expand[{{},{1,2}}])
  refined_box[{{},{1,2}}]:cmul(expand[{{},{1,2}}])
  refined_box[{{},{3,4}}]:add(expand[{{},{3,4}}])
  if Sub == true then refined_box[{{},{1,2}}]:add(expand[{{},{1,2}}]) end  
  refined_box = refined_box --+ real_box_ratio:view(1,4,20097):expand(n,4,20097)
  refined_box =refined_box:transpose(2,3)
--  local score, recognition = torch.max(conf,3)
  --net:float()
  -- nms
  for iter_image = 1, n do
    
    local image_name = list[iter_image+start_iter-1].image_name
    local size = image.load(image_name..'.jpg'):size()
    for iter_class =1, 20 do
       ::pass::
    local res = {}
--    local index = torch.eq(recognition[{iter_image,{},{}}],iter_class)
     
    local conf_image_class = conf[{iter_image,{},{iter_class}}]
    local index = torch.gt(conf_image_class,0.01)
 
    local detection_box = refined_box[iter_image]

    detection_box =detection_box[index:expandAs(detection_box)]
    if detection_box:numel() ==0 then goto pass end 
    detection_box = detection_box:view(-1,4)
--print(detection_box:size())
    local detection_score = conf_image_class[index]:view(-1)
    res.image_name = image_name
    
    --nms
    res.box, res.score = nms(detection_box,0.45,detection_score,size)
    
    write_txt(res, folder,iter_class)
    
    end

  end





  input_tensor =nil;
  collectgarbage();
  result = {}
  iter = end_iter + 1
  --net:cuda() 
  end

local i5 = os.time()
print('tot test time per an image',(i5-i1)/#list)
  collectgarbage()
  net:training()
  --net:cuda()


end

--map



function validation(net,savename,netname)

local valid_txt ='VOCdevkit/VOC2012/ImageSets/Main/val.txt'
local valid_list = {}
if paths.dirp('validation/'..netname )== false then os.execute('mkdir validation/'..netname) end

local valid_folder = 'validation/'..netname ..'/'
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
local n = 100
local randperm = torch.randperm(n)
rand = rand:index(1,randperm:long())
local new_list ={}
for iter = 1, n do
new_list[iter] = valid_list[rand[iter]]
end
-- new list write


local result = test(net,new_list,valid_folder..savename)


end
