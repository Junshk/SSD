require 'etc'
-- make ratio of box position in realimg to regress
-- prior is default box
-- iterative function bb, const


function prior_box(img_size,layer_size,min_max,aspect_ratios)  --xymm size for 500
if type(img_size) =='number' then img_size ={w=img_size,h=img_size}end
if type(layer_size) =='number' then layer_size ={w=layer_size,h=layer_size}end


local min_size = min_max.min
local max_size = -1


if min_max.max ~= nil then max_size = min_max.max end ; 
local mulmm = max_size*min_size



for iter = 1, #aspect_ratios do table.insert(aspect_ratios,1/aspect_ratios[iter]) end --flip
aspect_ratios[#aspect_ratios+1] =1


local num_priors  = #aspect_ratios
if max_size ~=-1 then num_priors= num_priors+1 end

top = torch.Tensor(4,num_priors,layer_size.h,layer_size.w)

-----------------bbox_w,h-------------------------------same as forward_cpu

local box_width = {}
local box_height = {}

box_width[1] =min_size; box_height[1]=min_size
if max_size~=-1 then table.insert(box_width,math.sqrt(mulmm)); 
table.insert(box_height,math.sqrt(mulmm)) end

for iter = 1, #aspect_ratios-1 do
table.insert(box_width,min_size*math.sqrt(aspect_ratios[iter]));
table.insert(box_height,min_size*math.sqrt(1/aspect_ratios[iter]));
end
box_width = torch.Tensor(box_width):view(num_priors,1,1):expand(num_priors,layer_size.h,layer_size.w)
box_height = torch.Tensor(box_height):view(num_priors,1,1):expand(num_priors,layer_size.h,layer_size.w)

--box_width = box_width:repeatTensor(layer_size.h,layer_size.w,1)
--box_height = box_height:repeatTensor(layer_size.h,layer_size.w,1)
---------------------bbox_cxy----------------------------------
local step_x = img_size.w / layer_size.w
local step_y = img_size.h / layer_size.h

local x_matrix = torch.range(1,layer_size.w):view(1,1,layer_size.w) ; 
x_matrix=x_matrix:expand(1,layer_size.h,layer_size.w)
local y_matrix = torch.range(1,layer_size.h):view(1,layer_size.h,1) ; 
y_matrix=y_matrix:expand(1,layer_size.h,layer_size.w)



x_matrix = x_matrix:expand(num_priors,layer_size.h,layer_size.w)
y_matrix = y_matrix:expand(num_priors,layer_size.h,layer_size.w)


local center_x = (x_matrix-0.5)*step_x; 
local center_y = (y_matrix-0.5)*step_y;
--print(center_x:size(),box_width:size(),box_height:size(),w_matrix:size()) 

top[{{1}}] = (center_x-box_width/2)
top[{{2}}] = (center_y-box_height/2)
top[{{3}}] = (center_x+box_width/2)
top[{{4}}] = (center_y+box_height/2)
top:clamp(0,img_size)

return top
end

-- example
--print(prior_box({w=500,h=500},{w=4,h=3},{min=475,max=555},{2,3}))
local t1 = prior_box(500,63,{min=35},{2})
local t2= prior_box(500,32,{max=155,min=75},{2,3})
local t3 = prior_box(500,16,{max=235,min=155},{2,3})
local t4 = prior_box(500,8,{max=315,min=235},{2,3})
local t5 = prior_box(500,4,{max=395,min=315},{2,3})
local t6 = prior_box(500,2,{max=475,min=395},{2,3})
local t7 = prior_box(500,1,{max=555,min=475},{2,3})

local function total_box(img_size) -- ratio or real size  image, xymin xymax
local div = 1
if img_size ~=nil then div = img_size end

return {t7/img_size, t6 /img_size,t5/img_size, t4/img_size, t3/img_size ,t2/img_size, t1/img_size}
end

function matching_gt_matrix(gt,img_size,iou) -- xmin, ymin, xmax, ymax

local box_size = total_box(img_size) --ratio

local size_default = 20097
local matched ={}
local match_tensor =torch.ByteTensor(1,size_default)
if iou == true then match_tensor = torch.Tensor(1,size_default)end
local idx =1

for iter = 1, #box_size do
local matching_for_tensor 
if iou == false then

  matching_for_tensor = torch.gt(jaccard_matrix(box_size[iter],gt),0.5)
elseif iou == true then 
--print(box_size[iter]:size())
--print('pboxgt',gt)

  matching_for_tensor = jaccard_matrix(box_size[iter],gt)
--print(matching_for_tensor)
end

local element = torch.numel(matching_for_tensor)

match_tensor[{{},{idx,idx+element-1}}] = matching_for_tensor:reshape(element)
idx = idx + element
--table.insert(matched,matching_for_tensor)
end


-- resize to 2d

return match_tensor--matched
end


function whcxy(img_size)
local whcxy = torch.Tensor(4,20097)
local xy = torch.Tensor(4,20097)
local real_size = total_box(img_size)

local idx =1
for iter =1 ,#real_size do
local element = torch.numel(real_size[iter])/4
local t = real_size[iter]:reshape(4,element)
xy[{{},{idx,idx+element-1}}] = t

idx =idx+ element
end

xy:clamp(0,1)
whcxy[{{1}}] =xy[{{3}}]-xy[{{1}}]
whcxy[{{2}}] = xy[{{4}}] -xy[{{2}}]
whcxy[{{3}}] = (xy[{{3}}]+xy[{{1}}])/2
whcxy[{{4}}] = (xy[{{4}}]+xy[{{2}}])/2

if prior_clip == true then whcxy:clamp(0,1) end

return whcxy
end

real_box_ratio = whcxy(500) --vectorize default box ratio

