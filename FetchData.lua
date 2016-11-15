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
math.randomseed(os.time())
-- choose aug type
::otherOpt::
local anno = anno_class[{{1,4},{}}]:clone()
local class = anno_class[{{5}}]:clone()

local w, h = img:size(3), img:size(2)
local aug_img
local anno_xy = torch.Tensor(anno:size())
local augType = math.random(3) ---------------------
local flip = math.random(2)

local function new_patch()  
  
  local crop_size = math.pow(0.1,math.random())
  local aspect = math.pow(2,math.random(-1,1))
  local crop_w, crop_h = math.min(math.floor(crop_size*math.sqrt(aspect)*w),w), 
                     math.min(math.floor(crop_size*math.sqrt(1/aspect)*h),h)
  local  crop_sx ,crop_sy = math.random(1,w-crop_w+1), math.random(1,h-crop_h+1)
  --math.ceil(w/2-math.random(1,math.floor(crop_w/2))),
  --math.ceil(h/2-math.random(1,math.floor(crop_h/2)))
   
  return  crop_w,crop_h,crop_sx, crop_sy 
end

  if augType == 1 then
  -- do nothing


  aug_img = img
  else 
    
    local  crop_w,crop_h,crop_sx, crop_sy = new_patch()
    
    local idx =1 
    if augType ==2 then 
    else
    repeat 
      if idx>50 then goto otherOpt end   
        math.randomseed(sys.clock()*10) 
           -- conform center of patch
        crop_w,crop_h,crop_sx, crop_sy = new_patch()

        local min_jaccard_ratio = math.random(5)/5-0.1--math.random(1,5)/5-0.1

        patch_window_ratio = torch.Tensor({crop_sx/w,crop_sy/h,(crop_sx+crop_w-1)/w,(crop_sy+crop_h-1)/h}):reshape(4,1)

        anno_xy[{{1}}] = (anno[{{3}}]-anno[{{1}}]/2)/w
        anno_xy[{{2}}] = (anno[{{4}}]-anno[{{2}}]/2)/h
        anno_xy[{{3}}] = (anno[{{3}}]+anno[{{1}}]/2)/w
        anno_xy[{{4}}] = (anno[{{4}}]+anno[{{2}}]/2)/h

        idx = idx+1
      until  torch.min(jaccard_matrix(anno_xy,patch_window_ratio))>min_jaccard_ratio
    end
      

      

      anno[{{3}}] = (anno[{{3}}]- crop_sx+1)--*ratio_width
      anno[{{4}}] = (anno[{{4}}]- crop_sy+1)--*ratio_height
      crop_w = math.min(w- crop_sx,crop_w)
      crop_h = math.min(h- crop_sy,crop_h)
      aug_img = image.crop(img,crop_sx,crop_sy,crop_sx+crop_w,crop_sy+crop_h)

  end

-----------------------------------------
-- anno to ratio
anno[{{1}}]:div(aug_img:size(3))
anno[{{2}}]:div(aug_img:size(2))
anno[{{3}}]:div(aug_img:size(3))
anno[{{4}}]:div(aug_img:size(2))
--print(anno)
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
--  print(xymin,xymax)
--  assert(torch.sum(torch.ge(xymin,xymax))==0) 
   anno[{{1,2}}] = xymax - xymin
   anno[{{3,4}}] = (xymax + xymin) /2
   anno:clamp(0,1)
   end
   --print(anno)
---flip
  assert(torch.sum(torch.eq(anno[{{2}}],0))==0)
  if flip == 1 then
    aug_img = image.hflip(aug_img)
    anno[{{3}}] = 1- anno[{{3}}]
  end

--scale to 500 by 500

aug_img = image.scale(aug_img,500,500)



return aug_img, anno, class
 

end

function dataload(ImgInfo) -- with normalize
math.randomseed(os.time())
::re::
--print('dataload')
local fetchNum = math.random(1,#ImgInfo) 


data = pascal_loadAImage({info = ImgInfo[fetchNum]})

local img = data.image[1]
local annoNum = #data.object[1]

local anno_class = torch.Tensor(5,annoNum)
 
for iter = 1, annoNum do

local anno = data.object[1][iter].bbox
local class = class2num(data.object[1][iter].class)

if anno[{1}]>=anno[{3}] or anno[{2}]>=anno[{4}] then assert(nil,"wrong anno"); goto re end

--anno = anno:cdiv(torch.Tensor({img:size(3),img:size(2),img:size(3),img:size(2)}))
--return whcxy form
anno_class[{{1},{iter}}] = (-anno[{1}]+anno[{3}])
anno_class[{{2},{iter}}] = (-anno[{2}]+anno[{4}])
anno_class[{{3},{iter}}] = (anno[{1}]+anno[{3}])/2
anno_class[{{4},{iter}}] = (anno[{2}]+anno[{4}])/2
anno_class[{{5},{iter}}] = class
--anno_class[{{},{iter}}] = torch.cat(anno,torch.Tensor({class}))

end

---input normalize
if bgr == true then
local vgg_img = torch.Tensor(img:size())

vgg_img[{{3}}] = (img[{{1}}]:float()-123)
vgg_img[{{2}}] = (img[{{2}}]:float()-117)
vgg_img[{{1}}] = (img[{{3}}]:float()-104)
img =vgg_img
elseif bgr ==false then
img[{{1}}] = (img[{{1}}]:float()-123)
img[{{2}}] = (img[{{2}}]:float()-117)
img[{{3}}] = (img[{{3}}]:float()-104)
end
---

img:div(norm)
--print(anno_class)
--- augmentation
--print(anno_class)
local aug_img,aug_anno,aug_class = augment(img,anno_class)

if aug_img == nil then goto re end


 assert(torch.sum(torch.eq(aug_anno[{{2}}],0))==0)
return aug_img, aug_anno, aug_class
end
-------------------------------------------------------------------------


local prior_whcxy = real_box_ratio:clone()

function make_default_anno(anno,class)--/////////////////// input cxy
--print('load make def')
local anno_default = prior_whcxy:clone()
local class_default = torch.Tensor(1,20097):fill(21)


local anno_n = anno:size(2)
local perm = torch.randperm(anno_n):long()
anno = anno:index(2,perm)
class = class:index(2,perm)

local iou_annos = torch.Tensor(anno_n,20097):fill(0)
local unused = torch.LongTensor(20097):fill(1)

for iter = 1, anno_n do

  local gt_xymm = torch.Tensor(4,1);

  gt_xymm[{1}]= -anno[{{1},{iter}}]/2 + anno[{{3},{iter}}]
  gt_xymm[{2}]= -anno[{{2},{iter}}]/2 + anno[{{4},{iter}}] 
  gt_xymm[{3}]= anno[{{1},{iter}}]/2 + anno[{{3},{iter}}]
  gt_xymm[{4}]= anno[{{2},{iter}}]/2 + anno[{{4},{iter}}]
   
  --local gt_class =class[{{1},{iter}}]:squeeze()

  iou_annos[{iter}] = matching_gt_matrix(gt_xymm,500,true)---// xymm

  --assert(gt_class<21 and gt_class >=1 ,'wrong class labeling '..gt_class)
  assert(torch.sum(torch.eq(anno[{{2},{iter}}],0))==0)
end

-- max
for iter = 1, anno_n do
  local v,id =torch.max(torch.cmul(iou_annos[iter],unused))
  anno_default[id] = anno[iter]
  class_default[id] = class[iter]
  unused[id] = 0 
end

-- rest
local v, id = torch.max(iou_annos,1)
local background = torch.lt(v,0.5)
usused:cmul(background)
anno_default[unused:view(1,-1):expand(anno_default)] = anno:index(1,id[unused])
class_default[unused] = class:index(1,id[unused])
--[[
  local expand_num = torch.sum(matching)
  anno_default[matching:expand(4,20097)] = anno[{{1,4},{iter}}]:expand(4,expand_num)---not perfect when overlap exist
  class_default[matching] = gt_class
]]--

end

return anno_default, class_default 
end

----------------------------------------------------------------------------
------------------------------------------------------------------------------
function patchFetch(batch_size,ImgInfo)
local default_size = 20097
local input_images = torch.Tensor(batch_size,3,500,500)
local target_anno =  torch.Tensor(batch_size,4,default_size)
local target_class = torch.Tensor(batch_size,1,default_size):fill(21)--1~21


for iter =1,batch_size do

local augmentedImg, aug_anno, aug_class= dataload(ImgInfo) -- for a image
-- default box matching !!
---- thanks to jihong,
 
  local anno_default, class_default = make_default_anno(aug_anno,aug_class)
  local mask = torch.eq((anno_default[{{2}}]),0) 

  if torch.sum(mask) >0 then print(anno_default[mask:expand(4,20097)]:view(4,-1));assert(nil) end

anno_default[{{3,4}}]:csub(prior_whcxy[{{3,4}}])
if Sub == true then anno_default[{{1,2}}]:csub(prior_whcxy[{{1,2}}]) end
anno_default[{{3,4}}]:cdiv(prior_whcxy[{{1,2}}])
anno_default[{{1,2}}]:cdiv(prior_whcxy[{{1,2}}])
--  print(anno_default[{{1,4},{1,4}}])
if logarithm == true then   
  anno_default[{{1,2}}]:log() 

  end
anno_default[{{1,2}}]:mul(var_w)
anno_default[{{3,4}}]:mul(var_x)

target_anno[{{iter}}] = anno_default -- - prior_whcxy-- w,h cx,cy
target_class[{{iter}}] = class_default
input_images[{{iter}}] = augmentedImg

end

local target = {}

target[1] = target_class;
target[2] = target_anno; --ratio 

--print(target[2][{{1,4},{},{1,4}}])
return input_images,target
end



