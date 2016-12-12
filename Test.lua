require 'cunn'
require 'cudnn'
require 'option'
require 'image'
require 'pascal'
require 'prior_box'
print('test load')
local batch = 4

torch.setdefaulttensortype('torch.FloatTensor')

local softmax = cudnn.LogSoftMax():cuda()
softmax:evaluate()

local img_save_iter = 1
--------------------------------------------
function write_txt(tot_result,folder,image_name,img)--,class_num)
  assert(folder~=nil, 'need folder name to write txt file')
   -- write in filename txt 
      if paths.dirp(folder) ==false then os.execute('mkdir '..folder) end

      --cutting 200 
      local bb_image
      if img ~= nil then bb_image = img 
      else bb_image = image.load(image_name.. '.jpg') end
            
      local image_id = image_name:split('/')-- string.sub(image_name,-11,-1)
            image_id =image_id[#image_id]
      
      if tot_result:numel() ==0 then goto save_result end 

      for class_num = 1, 20 do
        --local result = tot_result[class_num]
        --print(tot_result[{{},{6}}])
        local class_result = tot_result[torch.eq(tot_result[{{},{6}}],class_num):expand(tot_result:size())]
        if class_result:numel() == 0 then goto continue end
        class_result = class_result:view(-1,6)     

        local class_box , class_score = class_result[{{},{1,4}}],class_result[{{},{5}}]--result.box, result.score
        --if class_box:numel() ==0 then return end
        --local class_num = tot_result[{iter,6}]
        local write_result = io.open(folder..'/'..'comp3_det_test_'..num2class(class_num)..'.txt',"a")
        write_result:write('\n')
     
       -- print(class_box) 
--print(class_box:size(1),class_num)
            for iter2 = 1, class_box:size(1) do
            local box = class_box[{{iter2}}]:squeeze()
            local score = class_score[iter2]:squeeze()
            write_result:write(image_id,' ',score,' ',box[1],' ',box[2],' ',box[3],' ',box[4],'\n' )
        --    print('box',box)
            bb_image = image.drawRect(bb_image,(box[1]),(box[2]),(box[3]),(box[4]))
            local label = num2class(class_num)--string.format('%s_%f',num2class(class_num),score)
           -- bb_image = image.drawText(bb_image,label,math.max(box[1]-15,0),math.max(box[2]-15,0),{wrap=true})--,{size=5})
            end

        write_result:close() 
      ::continue::
      end

    ::save_result::
    image.save('conf/'..img_save_iter..'.jpg',bb_image)
    img_save_iter = img_save_iter+1
  
  bb_image = nil
  tot_result = nil
  collectgarbage()
  
  end

---------------------------------------------
function test_tensor(tensor,image_tensor,folder)
  conf = tensor[{{},{5}}]
  refined_box = tensor[{{},{1,4}}]
  refined_box[{{},{1,2}}]:div(var_w)
  refined_box[{{},{3,4}}]:div(var_x)
  local n = refined_box:size(1)
  local expand = real_box_ratio:view(1,20097,4):expand(n,20097,4)

  if logarithm == true then refined_box[{{},{},{1,2}}]:exp() end
  refined_box[{{},{},{3,4}}]:cmul(expand[{{},{},{1,2}}])
  refined_box[{{},{},{1,2}}]:cmul(expand[{{},{},{1,2}}])
  refined_box[{{},{},{3,4}}]:add(expand[{{},{},{3,4}}])
 --if Sub == true then refined_box[{{},{1,2}}]:add(expand[{{},{1,2}}]) end  
  --refined_box = refined_box --+ real_box_ratio:view(1,4,20097):expand(n,4,20097)
  --refined_box =refined_box:transpose(2,3)
  -- nms
  for iter_image = 1, n do
    
    local image_name = iter_image--list[iter_image+start_iter-1].image_name
    local size = image_tensor[{iter_image}]:size()--image.load(image_name..'.jpg'):size()
    
    local tot_output = torch.Tensor()
    
    for iter_class =1, 20 do
 --      ::pass::
    --print(iter_class)
    local res = {}
--    local index = torch.eq(recognition[{iter_image,{},{}}],iter_class)
     
    local conf_image_class = conf[{{iter_image}}]--conf[{iter_image,{},{iter_class}}]
    local index = torch.eq(conf_image_class,iter_class):squeeze():view(-1,1)
 
    local detection_box = refined_box[{iter_image}]
--print(detection_box:size(),index:size())
    detection_box =detection_box[index:expandAs(detection_box)]
    if detection_box:numel() ==0 then goto pass end 
    detection_box = detection_box:view(-1,4)
--print(detection_box:size())
    local detection_score = conf_image_class[index]:view(-1)
    --res.image_name = image_name
    --print(detection_box:numel()/4,iter_class)
    --nms
   -- res.box, res.score 
    local output = torch.Tensor(detection_box:size(1),6)
    --torch.cat(detection_box,detection_score,2)--nms(detection_box,0.45,detection_score,size)
    --print(output:size(), iter_class)
    output[{{},{1}}] = -detection_box[{{},{1}}]/2+detection_box[{{},{3}}]
    output[{{},{2}}] = -detection_box[{{},{2}}]/2+detection_box[{{},{4}}]
    output[{{},{3}}] = detection_box[{{},{1}}]/2+detection_box[{{},{3}}]
    output[{{},{4}}] = detection_box[{{},{2}}]/2+detection_box[{{},{4}}]

    output[{{},{1}}]:mul(size[3])
    output[{{},{2}}]:mul(size[2])
    output[{{},{3}}]:mul(size[3])
    output[{{},{4}}]:mul(size[2])
    
    output[{{},{5}}] = detection_score
    output[{{},{6}}] = iter_class
   -- print(detection_box)
    if tot_output:numel() ==0 then tot_output = output
    else tot_output = torch.cat({tot_output,output},1) end
   ::pass::     
    end
    
    -- discard wo 200 
    --local _,sort_idx  = tot_output[{{},5}]:sort(1,true)
    --tot_output = tot_output:index(1,sort_idx)
    
    write_txt(tot_output,folder,tostring(image_name),image_tensor[{iter_image}])--(res, folder,iter_class)
--::pass::
  end





end

function test(net,list,folder,opt)
  
  if pretrain == nil then
    pretrain = torch.load('pretrain.net')
  end
  pretrain:evaluate()

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
  
 -- local result_vector = torch.Tensor(#list,25,20097)

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

  --preprocess
    local img = image.load(imagename..'.jpg')
      img = image.scale(img,500,500)
      img = img*255
      img[{{1}}]:csub(r_mean)
      img[{{2}}]:csub(g_mean)
      img[{{3}}]:csub(b_mean)

      if bgr == true then 
        local vgg_img = torch.Tensor(img:size())
        vgg_img[{{1}}] = img[{{3}}]:clone()
        vgg_img[{{2}}] = img[{{2}}]:clone()
        vgg_img[{{3}}] = img[{{1}}]:clone()

        img = vgg_img:clone()
      end  

    input_tensor[{{iter-start_iter+1}}] = img --image.scale(img,500,500)
  
    end
          -----------
          --forward--
   local input_tensor_ = pretrain:forward(input_tensor:cuda())
   local output =net:forward(input_tensor_:cuda())--:float()
   local conf_before_softmax = output[1]--output[{{},{1,21}}]:transpose(2,3):reshape(n*20097,21)
  local conf =output[1]:float()-- softmax:forward(conf_before_softmax:cuda()):view(n,20097,21):exp():float()
  
  local refined_box = output[2]--output[{{},{22,25}}]

  refined_box[{{},{},{1,2}}]:div(var_w)
  refined_box[{{},{},{3,4}}]:div(var_x)

  local expand = real_box_ratio:view(1,20097,4):expand(n,20097,4):cuda()

  refined_box[{{},{},{1,2}}]:exp() 
  refined_box[{{},{},{3,4}}]:cmul(expand[{{},{},{1,2}}])
  refined_box[{{},{},{1,2}}]:cmul(expand[{{},{},{1,2}}])
  refined_box[{{},{},{3,4}}]:add(expand[{{},{},{3,4}}])
  --if Sub == true then refined_box[{{},{1,2}}]:add(expand[{{},{1,2}}]) end  
  --refined_box = refined_box --+ real_box_ratio:view(1,4,20097):expand(n,4,20097)
  --refined_box =refined_box:transpose(2,3)
  -- nms
  for iter_image = 1, n do
    
    local image_name = list[iter_image+start_iter-1].image_name
    local size = image.load(image_name..'.jpg'):size()
    
    local tot_output = torch.Tensor(201*20,6)
    local output_iter = 1
    local _, max_class = torch.max(conf[{iter_image,{},{1,20}}],conf[{iter_image}]:dim())
    local conf_image = nn.SoftMax():forward(conf[{iter_image}])
    assert(torch.sum(conf_image,2)[1]:squeeze()==1 ,'softmax sum')
    for iter_class =1, 20 do
      -- ::pass::
    --local res = {}
--    local index = torch.eq(recognition[{iter_image,{},{}}],iter_class)
     
    local conf_image_class = conf_image[{{},{iter_class}}]
    local index = torch.gt(conf_image_class,0.015)
    if opt ~=nil then index = torch.eq(max_class,iter_class) end
    if torch.sum(index) == 0 then goto pass end

    local detection_box = refined_box[iter_image]:float()

    detection_box =detection_box[index:expand(detection_box:size())]:view(-1,4)

    local detection_score = conf_image_class[index]:view(-1)
   -- res.image_name = image_name
    
    --nms
   -- res.box, res.score 
    local output = nms(detection_box,0.45,detection_score,size)
    
    if output:numel() ~= 0 then
    output[{{},{6}}] = iter_class
    tot_output[{{output_iter,output_iter+output:size(1)-1}}] = output
    output_iter = output_iter + output:size(1)
    end
   
    
    detection_box = nil
    detection_score = nil
    collectgarbage()
    ::pass:: 
        
    end -- iter class
   
    if output_iter == 1 then
      tot_output = torch.Tensor()
    else  
    tot_output = tot_output[{{1,output_iter-1}}]
    -- discard wo 200 
    local _,sort_idx  = tot_output[{{},5}]:sort(1,true)
  --  print(sort_idx)
  --  print(output_iter,sort_idx:size(),tot_output:size())
   -- print(math.max(output_iter-1,200))
    sort_idx = sort_idx[{{1,math.min(output_iter-1,200)}}]
    tot_output = tot_output:index(1,sort_idx)
    end

    write_txt(tot_output,folder,image_name)--(res, folder,iter_class)
    
  end





  input_tensor =nil;
  input_tensor_ =nil;
  refined_box = nil;
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

if paths.dirp('validation/'..netname )== false then os.execute('mkdir validation/'..netname) end
local valid_folder = 'validation/'..netname ..'/'



-- random sample list
local rand = torch.range(1,#valid_list)
local n = 100
local randperm = torch.randperm(n)
rand = rand:index(1,randperm:long())
local new_list ={}
for iter = 1, math.min(n, #valid_list) do
new_list[iter] = valid_list[rand[iter]]
end
-- new list write


local result = test(net,new_list,valid_folder..savename)--,true)


end
