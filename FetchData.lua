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
local anno = anno_class[{1,4}]
local class = anno_class[{5}]
local w, h = img:sie(3), img:size(2)
-- choose aug type
local augType = math.random(1,3)

  if augTYpe == 1 then
  -- do nothing
  else 
    local crop_size = math.pow(0.1,math.random())
    local aspect = math.pow(2,math.random(-1,1))
    local crop_w, crop_h = math.floor(crop_size*math.sqrt(aspect)*w), 
                           math.floor(crop_size*math.sqrt(1/aspect)*h)
    -- conform center of patch
    local crop_sx ,crop_sy = math.ceil(w/2-math.random(1,math.floor(crop_w/2))),0,
                             math.ceil(h/2-,math.random(1,math.floor(crop_h/2)))

    if augType == 2 then
            --do nothing

    elseif augType ==3 then
    -- jaccard ratio
      local min_jaccard_ratio = math.random(2,6)/6-0.1--math.random(1,5)/5-0.1
      local idx =0

      repeat 
      if idx>30 then goto otherOpt end
      patch_window_ratio = torch.Tensor({crop_sx/w,crop_sy/h,(crop_sx+crop_w-1)/w,(crop_sy+crop_h-1)/h}):reshape(4,1)
      anno_xy = torch.Tensor(anno)
      anno_xy[{1}] = anno[{3}]-anno[{1}]/2
      anno_xy[{2}] = anno[{4}]-anno[{2}]/2
      anno_xy[{3}] = anno[{3}]+anno[{1}]/2
      anno_xy[{4}] = anno[{4}]+anno[{2}]/2

      idx = idx+1
      until  torch.min(jaccard_matrix(anno_xy,patch))<min_jaccard_ratio


    end


    --cropping
      local ratio_width , ratio_height = w/crop_w, h/crop_h
      anno[{1}] = anno[{1}]*ratio_widht
      anno[{2}] = anno[{2}]*ratio_height
      anno[{3}] = (anno[{3}]- crop_sx/w)*ratio_width
      anno[{4}] = (anno[{4}]-crop_sy/h)*rartio_height

      img = image.crop(img,crop_sx,crop_sy,crop_sx+crop_w,crop_sy+crop+h)
  end



-- eliminate annotataion and its coherent class out of bound
local area_mask = torch.eq(torch.cmul(anno[{1}],anno[{2}]),0)
local out_mask = torch.gt(anno[{3}]-anno[{1}]/2,1) + torch.lt(anno[{3}]+anno[{1}]/2,0)
+ torch.gt(anno[{4}]-anno[{2}]/2,1) + torch.lt(anno[{4}]+anno[{2}]/2,0)

local rivive_mask = 1-area_mask:add(out_mask):clamp(0,1)
anno = anno[rivive_mask]
class = class[rivive_mask]

---flip
local flip = math.random(2)
  
  if flip == 1 then
    img = image.hflip(img)
    anno[{3}] = 1- anno[{3}]
  end


--scale to 500 by 500

  img = image.scale(img,500,500)

  return img, anno, class

end
--[[
function augment(img,gt_anno_class) --gt_anno : 4 by ...

  local anno_clone = gt_anno_class:clone()



  local w, h = img:size(3) , img:size(2)
--random
  ::otherOpt::
  local random = math.random(1,3)
  local random_aspect = math.pow(2,math.random(-1,1))
  local flip = math.random(1,2)
  local random_size =math.pow(0.1,math.random())
  local bg


-- augment_param

  local cx_ratio ,cy_ratio = (gt_anno_class[{1}]+gt_anno_class[{3}])/2, (gt_anno_class[{2}]+gt_anno_class[{4}])/2
  local crop_w, crop_h = math.floor(random_size*w*math.sqrt(random_aspect)), math.floor(random_size*h*math.sqrt(1/random_aspect))
  crop_w,crop_h = math.min(w,crop_w),math.min(h,crop_h)
  local sx,sy =math.random(1, w-crop_w+1),math.random(1,h-crop_h+1)


--------------------------------
  if random ==1 then

  else
    if random ==2 then
      local min_jaccard_ratio = math.random(2,6)/6-0.1--math.random(1,5)/5-0.1
      local patch = torch.Tensor({sx/w,sy/h,(sx+crop_w-1)/w,(sy+crop_h-1)/h})
      local idx =0

      repeat 
      if idx>30 then goto otherOpt end
      sx,sy=math.random(1, w-crop_w+1),math.random(1,h-crop_h+1)
patch = torch.Tensor({sx/w,sy/h,(sx+crop_w-1)/h,(sy+crop_h)/h}):reshape(4,1)

      idx = idx+1
      until  torch.min(jaccard_matrix(gt_anno_class[{{1,4}}],patch))<min_jaccard_ratio

   end

  anno_clone[{1}] = (gt_anno_class[{1}]-sx/w):cmax(0):cmin(crop_w/w)*w/crop_w
  anno_clone[{2}] = (gt_anno_class[{2}]-sy/h):cmax(0):cmin(crop_h/h)*h/crop_h
  anno_clone[{3}] = (gt_anno_class[{3}]-sx/w):cmin(crop_w/w):cmax(0)*w/crop_w
  anno_clone[{4}] = (gt_anno_class[{4}]-sy/h):cmin(crop_h/h):cmax(0)*h/crop_h

  bg =  torch.gt((cx_ratio*w-sx):cmul(cx_ratio*w-sx-crop_w+1),0)+ torch.gt((cy_ratio*h-sy):cmul(cy_ratio*h-sy-crop_h+1),0)+torch.eq(anno_clone[{1}],anno_clone[{3}])+torch.eq(anno_clone[{2}],anno_clone[{4}])

  bg:clamp(0,1)
-- remove gt out of bd


  local tf_fg = (1-bg:float()):view(1,gt_anno_class:size(2))
  tf_fg = tf_fg:expandAs(gt_anno_class)


  anno_clone = (anno_clone[tf_fg:byte()])
  if anno_clone:dim() ==0 then return nil end


  local element_anno = torch.numel(anno_clone)
  anno_clone = anno_clone:view(5,element_anno/5)



  img = image.crop(img,sx,sy,sx+crop_w-1,sy+crop_h-1)


end
------------------------------



img = image.scale(img,500,500) -- anno not changed

-- compute anno of img



local class_ = torch.Tensor(1,20097):fill(21)
local anno_ = torch.Tensor(4,20097):fill(0)
anno_clone[{{1,4}}]:clamp(0,1)



if flip==1 then 
img = image.hflip(img)
anno_clone[{{1,4}}] = 1- anno_clone[{{1,4}}]

anno_clone = anno_clone:index(1,torch.LongTensor{3,4,1,2,5})
end
-- annotate class num

if torch.sum(torch.eq(anno_clone[{1}],anno_clone[{3}])+torch.sum(torch.eq(anno_clone[{2}],anno_clone[{4}])))~=0 then
        print( 'wrong anno_clone',anno_clone,bg,gt_anno_class,random,flip  ); assert(nil) end
local anno_n =anno_clone:size(2)

for iter = 1, anno_n do
  local gt_iter = anno_clone[{{1,4},{iter}}]:squeeze()
  local gt_class =anno_clone[{{5},{iter}}]:squeeze()

  local matching = matching_gt_matrix(gt_iter,500)
  assert(gt_class<21 and gt_class >=1 ,'wrong class labeling '..gt_class)
  class_[matching] = gt_class
  gt_iter =gt_iter:squeeze()

  local whcxy = torch.Tensor({gt_iter[{3}]-gt_iter[{1}],gt_iter[{4}]-gt_iter[{2}],(gt_iter[{1}]+gt_iter[{3}])/2,(gt_iter[{2}]+gt_iter[{4}])/2}):reshape(4,1)

  local expand_num = torch.sum(matching)
  anno_[matching:expand(4,20097)] = whcxy:expand(4,expand_num)--////////////////////
end


  return img, anno_, class_
end
--------------------------------------------------------
]]--
function dataload(ImgInfo) -- with normalize
::re::

local fetchNum = math.random(1,#ImgInfo) 


data = pascal_loadAImage({info = ImgInfo[fetchNum]})

local img = data.image[1]
local annoNum = #data.object[1]

local anno_class = torch.Tensor(5,annoNum)

for iter = 1, annoNum do

local anno = data.object[1][iter].bbox
local class = class2num(data.object[1][iter].class)

if anno[{1}]>=anno[{3}] or anno[{2}]>=anno[{4}] then goto re end

--anno = anno:cdiv(torch.Tensor({img:size(3),img:size(2),img:size(3),img:size(2)}))
--return whcxy form
anno_class[{{1},{iter}}] = (-anno[{1}]+anno[{3}])
anno_class[{{2},{iter}}] = (-anno[{2}]+anno[{4}])
anno_class[{{3},{iter}}] = (anno[{1}]+anno[{3}])/2
anno_class[{{4},{iter}}] =(anno[{1}]+anno[{3}])/2
anno_class[{{5},{iter}}] =class
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

--- augmentation
local aug_img,aug_anno,aug_class = augment(img,anno_class)

if aug_img == nil then goto re end


return aug_img, aug_anno, aug_class
end
-------------------------------------------------------------------------


local prior_whcxy = real_box_ratio:clone()

function patchFetch(batch_size,ImgInfo)
local default_size = 20097
local input_images = torch.Tensor(batch_size,3,500,500)
local target_anno =  torch.Tensor(batch_size,4,default_size):fill(0) 
local target_class = torch.Tensor(batch_size,1,default_size):fill(0)--1~21


for iter =1,batch_size do

local augmentedImg,aug_anno, aug_class= dataload(ImgInfo) -- for a image

input_images[{{iter}}] = augmentedImg
-- default box matching !!
---- thanks to jihong,
print(aug_anno[{{},{1}}])
aug_anno[{{3,4}}]:csub(prior_whcxy[{{3,4}}])
aug_anno[{{3,4}}]:cdiv(prior_whcxy[{{1,2}}])
aug_anno[{{1,2}}]:cdiv(prior_whcxy[{{1,2}}])
if logarithm == true then aug_anno[{{1,2}}]:log() end
aug_anno[{{1,2}}]:mul(var_w)
aug_anno[{{3,4}}]:mul(var_x)

print(aug_anno[{{},{1}}])
target_anno[{{iter}}] = aug_anno -- - prior_whcxy-- w,h cx,cy
target_class[{{iter}}] = aug_class

end


local target = {}

target[1] = target_class;
target[2] = target_anno; --ratio 

return input_images,target
end



