require 'image'
require 'etc'
require 'pascal'



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

local anno_class = {}

for iter = 1, annoNum do

local anno = data.object[1][iter].bbox
local class = class2num(data.object[1][iter].class)

table.insert(anno_class,torch.cat(anno,torch.Tensor({class})))

end


return img:float()/255, anno_class
end


function augment(img,gt_anno)

local random = math.random(1,3)
local random_aspect = math.pow(2,math.random(-1,1))
local flip = math.random(1,2)
local random_size =math.pow(0.1,math.random())*500
local sx,sy 

local w, h = math.floor(random_size*math.sqrt(random_aspect)), math.floor(random_size*math.sqrt(1/random_aspect))
w,h = math.min(500,w),math.min(h,500)


if random ==1 then
elseif random ==2 then

jaccard()
elseif random ==3 then

sx,sy =math.random(1, 500-w+1),math.random(1,500-h+1)

img = image.crop(img,sx,sy,sx+w-1,sy+h-1)
end



img = image.scale(img,500,500)


if flip==1 then 
img = image.hflip(img)

end


-- compute anno of img

return img, anno
end

function patchFetch(batch_size,ImgInfo)

local input_batch = torch.Tensor(batch_size,3,500,500)
local input_anno = torch.Tensor(batch_size,default,4):fill(0) 
local input_class = torch.Tensor(batch_size,default,1):fill(0) --1~21


for iter =1,batch_size do
local img, anno_class =  dataload(ImgInfo) -- default by 5
local augmentedImg,augmentedAnno = augment(img,anno_class) -- for a image
input_batch[{{iter}}] = augmentedImg
input_anno[{{iter}}] = augmentedAnno
input_class[{{iter}}] = anno_class[{{},{5}}]
end


local target = {}

target[1] = input_class;
target[2] = input_anno; --ratio 

return input_batch,target
end



