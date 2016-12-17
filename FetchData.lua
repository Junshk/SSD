require 'image'
--require 'etc'
require 'pascal'
require 'prior_box'
--require 'nn'

torch.setdefaulttensortype('torch.FloatTensor')
torch.setnumthreads(2)
--------------------------------------------
--------------------------------------------------------------
function augment(img,anno_class)
--if random_data2 ~=true then 
  math.randomseed(sys.clock())--os.time())
--end
-- choose aug type
::otherOpt::
local anno = anno_class[{{1,4},{}}]:clone()
local class = anno_class[{{5}}]:clone()

local w, h = img:size(3), img:size(2)
local aug_img
local anno_xy = torch.Tensor(anno:size())
local augType = math.random(3)--------------------
local flip = math.random(2)

local function new_patch()  
  math.randomseed(os.time())  
  local crop_size = 0.1+(1-0.1)*math.random()
  local aspect = math.pow(2,math.random(-1,1))
  local crop_w, crop_h = math.floor(crop_size*math.sqrt(aspect)*w), 
                     math.floor(crop_size*math.sqrt(1/aspect)*h)
  local  crop_sx ,crop_sy = math.random(0,math.max(w-crop_w,1)), math.random(0,math.max(h-crop_h,1))
   
  return  crop_w,crop_h,crop_sx, crop_sy 
end





  if augType == 1 then
  -- do nothing


  aug_img = img
  
  -- additional aug
  --aug_img = aug_img * math.random(0.9,1.1)
  --aug_img:clamp(0,255)

  else 
   math.randomseed(sys.clock()) 
    local  crop_w,crop_h,crop_sx, crop_sy = new_patch()
    
    local idx =1 
    if augType ==2 then 
    else
    repeat 
   if idx>30 then goto otherOpt end   
        math.randomseed(sys.clock()*100) 
           -- conform center of patch
        crop_w,crop_h,crop_sx, crop_sy = new_patch()

        local min_jaccard_ratio = math.random(5)/5-0.1--math.random(1,5)/5-0.1

        patch_window_ratio = torch.Tensor({crop_sx/w,crop_sy/h,(crop_sx+crop_w)/w,(crop_sy+crop_h)/h}):reshape(4,1)

        anno_xy[{{1}}] = (anno[{{3}}]-anno[{{1}}]/2)/w
        anno_xy[{{2}}] = (anno[{{4}}]-anno[{{2}}]/2)/h
        anno_xy[{{3}}] = (anno[{{3}}]+anno[{{1}}]/2)/w
        anno_xy[{{4}}] = (anno[{{4}}]+anno[{{2}}]/2)/h

        idx = idx+1
      until  torch.min(jaccard_matrix(anno_xy,patch_window_ratio))>min_jaccard_ratio
    end
      

      

      anno[{{3}}] = (anno[{{3}}]- crop_sx)--*ratio_width
      anno[{{4}}] = (anno[{{4}}]- crop_sy)--*ratio_height
      local r_crop_w = math.min(w- crop_sx,crop_w)
      local r_crop_h = math.min(h- crop_sy,crop_h)
      --aug_img = torch.Tensor(3,crop_h,crop_w):fill(0)
      --[[
      if chm == true then 
        aug_img[{{1}}]:fill(r_mean)
        aug_img[{{2}}]:fill(g_mean)
        aug_img[{{3}}]:fill(b_mean)
      end  
      aug_img:div(norm)
      ]]--
      --aug_img[{{},{1,r_crop_h},{1,r_crop_w}}] 
      aug_img
      = image.crop(img,crop_sx,crop_sy,crop_sx+r_crop_w,crop_sy+r_crop_h)

  end

-----------------------------------------
--print('in aug',anno)
-- anno to ratio
anno[{{1}}]:div(aug_img:size(3))
anno[{{2}}]:div(aug_img:size(2))
anno[{{3}}]:div(aug_img:size(3))
anno[{{4}}]:div(aug_img:size(2))
   local center_mask = torch.gt(anno[{{3}}],1) + torch.lt(anno[{{3}}],0) + torch.gt(anno[{{4}}],1) + torch.lt(anno[{{4}}],0)
   center_mask = 1- center_mask:clamp(0,1)

   anno = anno[center_mask:expand(anno:size())]
   class = class[center_mask]
   if anno:numel() ==0 then  --print('oopt');
   goto otherOpt; end
  assert(torch.sum(torch.lt(anno,0))==0) 
   anno = anno:view(4,-1)
   class = class:view(1,-1)
   
   if truck == true then
   local xymin, xymax = anno[{{3,4}}]- anno[{{1,2}}]/2, anno[{{3,4}}]+anno[{{1,2}}]/2

  xymin:clamp(0,1)
   xymax:clamp(0,1)
   anno[{{1,2}}] = xymax - xymin
   anno[{{3,4}}] = (xymax + xymin) /2
   end

---flip
  assert(torch.sum(torch.eq(anno[{{2}}],0))==0)
  if flip == 1 then
    aug_img = image.hflip(aug_img)
    anno[{{3}}] = 1- anno[{{3}}]
  end



--scale to 500 by 500

aug_img = image.scale(aug_img,500,500)

t_num =1
while paths.filep('conf/gt'..t_num..'.jpg') ==true do
t_num = t_num+1
end
local bb_img = aug_img:clone()

for iter = 1, class:numel() do
local x_,y_,xu,yu = math.max(1,math.ceil(500*(anno[{3,iter}]-anno[{1,iter}]/2))),math.max(1,math.ceil(500*(anno[{4,iter}]-anno[{2,iter}]/2))),math.ceil(500*(anno[{3,iter}]+anno[{1,iter}]/2)),math.ceil(500*(anno[{4,iter}]+anno[{2,iter}]/2))
--print(x_,y_,xu,yu)
bb_img = image.drawRect(bb_img,x_,y_,xu,yu)
bb_img = image.drawText(bb_img,num2class(class[{1,iter}]),x_,y_)
end
if t_num < 100 then
image.save('conf/gt'..t_num..'.jpg',bb_img/255)
end
--print('AG',augType)
return aug_img, anno, class
 

end

function dataload(ImgInfo,num) -- with normalize
--math.randomseed(sys.clock())
::re::
local fetchNum = num or math.random(1,#ImgInfo) 


data = pascal_loadAImage({info = ImgInfo[fetchNum]})

local img = data.image[1]
local annoNum = #data.object[1]

local anno_class = torch.Tensor(5,annoNum)
 
for iter = 1, annoNum do

local anno = data.object[1][iter].bbox
local class = class2num(data.object[1][iter].class)

if anno[{1}]>=anno[{3}] or anno[{2}]>=anno[{4}] then assert(nil,"wrong anno"); goto re end

--return whcxy form
anno_class[{{1},{iter}}] = (-anno[{1}]+anno[{3}])
anno_class[{{2},{iter}}] = (-anno[{2}]+anno[{4}])
anno_class[{{3},{iter}}] = (anno[{1}]+anno[{3}])/2
anno_class[{{4},{iter}}] = (anno[{2}]+anno[{4}])/2
anno_class[{{5},{iter}}] = class
--anno_class[{{},{iter}}] = torch.cat(anno,torch.Tensor({class}))

end

--- augmentation
local aug_img,aug_anno,aug_class = augment(img,anno_class)

if aug_img == nil then goto re end

---input normalize
if chm == true then
aug_img[{{1}}]:csub(r_mean)
aug_img[{{2}}]:csub(g_mean)
aug_img[{{3}}]:csub(b_mean)
--print(img)
end
if bgr == true then
local vgg_img = torch.Tensor(aug_img:size())

vgg_img[{{3}}] = (aug_img[{{1}}]):clone()--:float())
vgg_img[{{2}}] = (aug_img[{{2}}]):clone()--:float())
vgg_img[{{1}}] = (aug_img[{{3}}]):clone()--:float())
aug_img =vgg_img:clone()
elseif bgr ==false then
end
---

  aug_img:div(norm)

  --print(aug_anno,aug_class)
  assert(torch.sum(torch.eq(aug_anno[{{2}}],0))==0)
  return aug_img, aug_anno, aug_class

end
-------------------------------------------------------------------------


local prior_whcxy = real_box_ratio:clone()

function make_default_anno(anno,class)--/////////////////// input cxy
--print('p',anno)
local anno_default = torch.Tensor(20097,4):copy(real_box_ratio)
local class_default = torch.Tensor(20097,1):fill(21)
local gt_xymm = torch.Tensor(anno:size())
  gt_xymm[{{1}}]= -anno[{{1}}]/2 + anno[{{3}}]
  gt_xymm[{{2}}]= -anno[{{2}}]/2 + anno[{{4}}] 
  gt_xymm[{{3}}]= anno[{{1}}]/2 + anno[{{3}}]
  gt_xymm[{{4}}]= anno[{{2}}]/2 + anno[{{4}}]
--print(gt_xymm) 
--assert(nil)
local anno_n = anno:size(2)
--local perm = torch.randperm(anno_n):long()
--anno = anno:index(2,perm)
--class = class:index(2,perm)

local iou_annos = torch.Tensor(anno_n,20097):fill(0)
local unused = torch.ByteTensor(20097):fill(1)

for iter = 1, anno_n do

  local gt_xymm_iter = gt_xymm[{{},iter}]:clone();
 
 --print('GTXYMM',gt_xymm_iter) 
   iou_annos[{iter}] = multi_jaccard(xy_box_ratio,gt_xymm_iter)-- matching_gt_matrix(gt_xymm_iter,500,true)---// xymm
   assert(torch.sum(torch.eq(anno[{{2},{iter}}],0))==0)
end

--print('a',anno[1])
-- CLIPPING ANNO
gt_xymm:clamp(0,1)
anno[{{1}}] = gt_xymm[{{3}}] - gt_xymm[{{1}}]
anno[{{2}}] = gt_xymm[{{4}}] - gt_xymm[{{2}}]
anno[{{3}}] = gt_xymm[{{1}}]/2+gt_xymm[{{3}}]/2
anno[{{4}}] = gt_xymm[{{2}}]/2+gt_xymm[{{4}}]/2
--print(anno[1])
-- max


for iter = 1, anno_n do
  local iter_anno = iou_annos[iter] + unused:float()
    local v,id =torch.max(iter_anno,1)
 -- print(torch.sum(torch.ne(iter_anno,1)))
 -- print(iter_anno[torch.ne(iter_anno,1)])
  id = id:squeeze()
  --print('id',id,v)
  anno_default[{id}] = anno[{{},iter}]
  class_default[{id}] = class[{{},iter}]
  unused[id] = 0 
end

-- rest
local v, id = torch.max(iou_annos,1)
local foreground = torch.gt(v,0.5)
unused:cmul(foreground)  
--id = id[unused]
id =id:squeeze()

for iter = 1,(unused):size(1) do
  if unused[iter] == 1 then
--print(anno:size(),id:size())
anno_default[{iter}] =anno[{{},{ id[iter]}}]--unused:view(1,-1):expand(anno_default:size())] =
--anno:index(2,id:long())
class_default[{iter}] = class[{{},{id[iter]}}]--unused] = class:index(2,id)
  end
end


 

return anno_default, class_default 
end

----------------------------------------------------------------------------
------------------------------------------------------------------------------
function patchFetch(batch_size,ImgInfo,seed)

local default_size = 20097
local input_images = torch.Tensor(batch_size,3,500,500)
local target =  {} --torch.Tensor(batch_size,default_size)
target[1] = torch.Tensor(batch_size,20097,1)
target[2] = torch.Tensor(batch_size,20097,4)  

torch.manualSeed((seed or 0)+ os.time())
local nums = torch.randperm(#ImgInfo)

for iter =1,batch_size do
  local num = nums[iter]
  local augmentedImg, aug_anno, aug_class= dataload(ImgInfo,num) -- for a image
-- default box matching !!
---- thanks to jihong,
assert(torch.sum(torch.eq(aug_anno,math.huge))==0) 
  local anno_default, class_default = make_default_anno(aug_anno,aug_class)
assert(torch.sum(torch.eq(anno_default,math.huge))==0) 
  --local mask = torch.eq((anno_default[{{},{2}}]),0) 
--------------------------------------------------
  --if torch.sum(mask) >0 then print(anno_default[mask:expand(20097,4)]:view(-1,4));assert(nil) end
assert(torch.sum(torch.eq(anno_default[{{},{1,2}}],0))==0)
anno_default[{{},{3,4}}]:csub(prior_whcxy[{{},{3,4}}])
anno_default[{{},{3,4}}]:cdiv(prior_whcxy[{{},{1,2}}])
anno_default[{{},{1,2}}]:cdiv(prior_whcxy[{{},{1,2}}])
  
anno_default[{{},{1,2}}]:log() 


anno_default[{{},{1,2}}]:mul(var_w)
anno_default[{{},{3,4}}]:mul(var_x)
----------------------------------------------
assert(torch.sum(torch.eq(anno_default,math.huge))==0) 

target[2][{{iter}}] = anno_default
target[1][{{iter}}] = class_default
input_images[{{iter}}] = augmentedImg

end

return input_images,target
end



