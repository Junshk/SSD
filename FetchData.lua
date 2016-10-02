require 'image'
require 'etc'
require 'pascal'
require 'prior_box'

torch.setdeaulttensortype('torch.FloatTensor')
function dataload(ImgInfo) -- with normalize

--[[
local num = #data.name

local fetchNum = math.random(1,num) 

local img = data.image[fetchNum]
local annoNum = #data.object[fetchNum]

local anno_class = {}

for iter = 1, annoNum do

local anno = data.object[fetchNum][iter].bbox
local class = class2num(data.object[fetchNum][iter].class)

table.insert(anno_class,torch.concat(anno,class))

end
]]--

local fetchNum = math.random(1,#ImgInfo) 


data = pascal_loadAImage({info = ImgInfo[fetchNum]})
local img = data.image[1]
local annoNum = #data.object[1]

local anno_class = torch.Tensor(5,annoNum)

for iter = 1, annoNum do

local anno = data.object[1][iter].bbox
local class = class2num(data.object[1][iter].class)

anno = anno:cdiv(torch.Tensor({img:size(3),img:size(2),img:size(3),img:size(2)}))

anno_class[{{},{iter}}] = torch.cat(anno,torch.Tensor({class}))

end


return img:float()/255, anno_class
end
-------------------------------------------------------------------------

function augment(img,gt_anno_class) --gt_anno : 4 by ...



--local anno = gt_anno_class[{{1,4}}]:clone()

local w, h = img:size(3) , img:size(2)
--random
::otherOpt::
local random = math.random(1,3)
local random_aspect = math.pow(2,math.random(-1,1))
local flip = math.random(1,2)
local random_size =math.pow(0.1,math.random())

--local bg_class = torch.ByteTensor(gt_anno_class:size(2)):fill(0)

-- augment_param

local cx_ratio ,cy_ratio = (gt_anno_class[{1}]+gt_anno_class[{3}])/2, (gt_anno_clas[{2}]+gt_anno_class[{4}])/2
local crop_w, crop_h = math.floor(random_size*w*math.sqrt(random_aspect)), math.floor(random_size*h*math.sqrt(1/random_aspect))
crop_w,crop_h = math.min(w,crop_w),math.min(h,crop_h)
local sx,sy =math.random(1, w-crop_w+1),math.random(1,h-crop_h+1)


--------------------------------
if random ==1 then
elseif random ==2 then
local min_jaccard_ratio = math.random(1,5)/5-0.1
local patch = torch.Tensor({sx/w,sy/h,(sx+crop_w-1)/w,(sy+crop_h-1)/h})
local idx =0

repeat 
if idx>20 then goto otherOpt end
sx,sy=math.random(1, w-crop_w+1),math.random(1,h-crop_h+1)
patch = torch.Tensor({sx/w,sy/h,(sx+crop_w-1)/h,(sy+crop_h)/h})

idx = idx+1
until  torch.min(jaccard(patch,gt_anno_class:t()))<min_jaccard_ratio


elseif random ==3 then

img = image.crop(img,sx,sy,sx+w-1,sy+h-1)

end
------------------------------
gt_anno_class[{1}] = (gt_anno_class[{1}]-sx/w):cmax(0)*w/crop_w
gt_anno_class[{2}] = (gt_anno_class[{2}]-sy/h):cmax(0)*h/crop_h
gt_anno_class[{3}] = (gt_anno_class[{3}]-sx/w):cmin(crop_w/w)*w/crop_w
gt_anno_class[{4}] = (gt_anno_class[{4}]-sy/h):cmin(crop_h/h)*h/crop_h

local bg =  torch.gt((cx_ratio*w-sx):cmul(cx_ratio*w-sx-crop_w+1),0)+ torch.gt((cy_ratio*h-sy):cmul(cy_ratio*h-sy-crop_h+1),0)

bg:clamp(0,1)
-- remove gt out of bd
gt_anno_class = (gt_anno_class:t()[(1-bg):byte()]):t()
-------------
local preSize = img:size()
img = image.scale(img,500,500) -- anno not changed

if flip==1 then 
img = image.hflip(img)
gt_anno_class[{{1,4}}] = 1 - gt_anno_class[{{1,4}}]
end

-- compute anno of img

gt_anno_class[{{1,4}}]:clamp(0,1)
--gt_anno_class[{5}][bg] = 21


local class_ = torch.Tensor(1,20097)
local anno_ = torch.Tensor(20097,4)
-- annotate class num
for iter = 1, gt_anno_class:size(2) do
local matching = matching_gt_matrix(gt_anno_class[{{1,4},{iter}}])
class_[matching] = gt_anno_class[{{5}}]
anno_[matching] = gt_anno_class[{{1,4}}]
end

return img, anno_:t(), class_
end


function patchFetch(batch_size,ImgInfo)
local default_size = 20097

local input_images = torch.Tensor(batch_size,3,500,500)
local target_anno =  torch.Tensor(batch_size,4,default_size):fill(0) 
local target_class = torch.Tensor(batch_size,1,default_size)fill(0)--1~21


for iter =1,batch_size do
local img, anno_class =  dataload(ImgInfo) -- default by 5
local augmentedImg,augmentedAnno, target_class[{{iter}}]
= augment(img,anno_class) -- for a image

input_images[{{iter}}] = augmentedImg

-- default box matching !!

target_anno[{{iter}}] = augmentedAnno

end


local target = {}

target[1] = target_class;
target[2] = target_anno; --ratio 

return input_images,target
end



