require 'etc'
require 'option'
-- make ratio of box position in realimg to regress
-- prior is default box
-- iterative function bb, const


local function prior_box(img_size,layer_size,min_max,aspect_ratios)  --xymm size for 500
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

top[{{1}}]:copy(center_x-box_width/2):clamp(0,img_size.w)
top[{{2}}]:copy(center_y-box_height/2):clamp(0,img_size.h)
top[{{3}}]:copy(center_x+box_width/2):clamp(0,img_size.w)
top[{{4}}]:copy(center_y+box_height/2):clamp(0,img_size.h)

return top
end

-- example
--print(prior_box({w=500,h=500},{w=4,h=3},{min=475,max=555},{2,3}))


function box_to_tensor(box_tensor)
require 'nn'
if box_tensor:dim() ~= 4 then assert(nil,box_tensor:dim()) end
local map_size = box_tensor:size(box_tensor:dim())
local n = nn.Sequential()
--n:add(nn.Reshape(-1,map_size,map_size,false))
n:add(nn.Transpose({1,2},{2,3},{3,4}))
--n:add(nn.Transpose({1,2},{2,3}))
n:add(nn.View(-1,4):setNumInputDims(4))

return n:forward(box_tensor)
end

local function total_box(img_size) -- ratio or real size  image, xymin xymax
local div = 1
if img_size ~=nil then div = img_size end


local t1 = box_to_tensor(prior_box(img_size,63,{min=50},{2}))
local t2= box_to_tensor(prior_box(img_size,32,{max=170,min=100},{2,3}))
local t3 = box_to_tensor(prior_box(img_size,16,{max=240,min=170},{2,3}))
local t4 = box_to_tensor(prior_box(img_size,8,{max=310,min=240},{2,3}))
local t5 = box_to_tensor(prior_box(img_size,4,{max=380,min=310},{2,3}))
local t6 = box_to_tensor(prior_box(img_size,2,{max=450,min=380},{2,3}))
local t7 = box_to_tensor(prior_box(img_size,1,{max=520,min=450},{2,3}))

return torch.cat({t7/img_size, t6 /img_size,t5/img_size, t4/img_size, t3/img_size ,t2/img_size, t1/img_size},1)
end
--[[
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
]]--

function whcxy(img_size)

local whcxy = torch.Tensor(20097,4)
local xy = torch.Tensor(20097,4)
local real_size = total_box(img_size)


xy:copy(real_size )
whcxy[{{},{1}}] =xy[{{},{3}}]-xy[{{},{1}}]
whcxy[{{},{2}}] = xy[{{},{4}}] -xy[{{},{2}}]
whcxy[{{},{3}}] = (xy[{{},{3}}]+xy[{{},{1}}])/2
whcxy[{{},{4}}] = (xy[{{},{4}}]+xy[{{},{2}}])/2

if prior_clip == true then whcxy:clamp(0,1) end
--whcxy = whcxy:t():contiguous()
--print(whcxy)
return whcxy
end
xy_box_ratio = total_box(image_size)
real_box_ratio = whcxy(image_size)--vectorize default box ratio
