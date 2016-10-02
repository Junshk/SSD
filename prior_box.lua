require 'etc'
-- make ratio of box position in realimg to regress
-- prior is default box
-- iterative function bb, const


function prior_box(img_size,layer_size,min_max,aspect_ratios)
if type(img_size) =='number' then img_size ={w=img_size,h=img_size}end
if type(layer_size) =='number' then layer_size ={w=layer_size,h=layer_size}end


local min_size = min_max.min
local max_size = -1
--print('mm',min_max)
--print(min_size)
if min_max.max ~= nil then max_size = min_max.max end ; 
local mulmm = max_size*min_size



for iter = 1, #aspect_ratios do table.insert(aspect_ratios,1/aspect_ratios[iter]) end --flip
aspect_ratios[#aspect_ratios+1] =1


local num_priors  = #aspect_ratios
if max_size ~=-1 then num_priors= num_priors+1 end

top = torch.Tensor(4,num_priors,layer_size.h,layer_size.w)

------------------------------------------------same as forward_cpu
local step_x = img_size.w / layer_size.w
local step_y = img_size.h / layer_size.h

local dim = layer_size.h * layer_size.w * num_priors *4 -- 4:xmin ymin xmax ymax

local w_matrix = torch.range(1,layer_size.w) ; w_matrix=w_matrix:repeatTensor(layer_size.h,1)
local h_matrix = torch.range(1,layer_size.h) ; h_matrix=h_matrix:repeatTensor(layer_size.w,1):t()
--print(w_matrix,h_matrix)
--w_matrix= w_matrix:reshape(layer_size.h*layer_size.w)
--h_matrix= h_matrix:reshape(layer_size.h*layer_size.h)
w_matrix = w_matrix:repeatTensor(num_priors,1,1)
h_matrix = h_matrix:repeatTensor(num_priors,1,1)
--print(w_matrix)
local box_width = {}
local box_height = {}

box_width[1] =min_size; box_height[1]=min_size
if max_size~=-1 then table.insert(box_width,math.sqrt(mulmm)); 
table.insert(box_height,math.sqrt(mulmm)) end

for iter = 1, #aspect_ratios-1 do
table.insert(box_width,min_size*math.sqrt(aspect_ratios[iter]));
table.insert(box_height,min_size*math.sqrt(1/aspect_ratios[iter]));
end

box_width = torch.Tensor(box_width)
box_height = torch.Tensor(box_height)
--print(box_height:size())
box_width = box_width:repeatTensor(layer_size.h,layer_size.w,1)
box_height = box_height:repeatTensor(layer_size.h,layer_size.w,1)
--print(box_width:size())

--local center_x = torch.Tensor(w_matrix:size())
--local center_y = torch.Tensor(h_matrix:size())

local center_x = (w_matrix+0.5)*step_x; local center_y = (h_matrix+0.5)*step_y;
--print(center_x:size(),box_width:size(),box_height:size(),w_matrix:size()) 

top[{{1}}] = (center_x-box_width/2)/img_size.w
top[{{2}}] = (center_y-box_height/2)/img_size.h
top[{{3}}] = (center_x+box_width/2)/img_size.w
top[{{4}}] = (center_y+box_height/2)/img_size.h

top:clamp(0,1)
return top
end

-- example
--print(prior_box({w=500,h=500},{w=4,h=3},{min=475,max=555},{2,3}))

function total_box()


local t1 = prior_box(500,63,{min=35},{2})
local t2= prior_box(500,32,{max=155,min=75},{2,3})
local t3 = prior_box(500,16,{max=235,min=155},{2,3})
local t4 = prior_box(500,8,{max=315,min=235},{2,3})
local t5 = prior_box(500,4,{max=395,min=315},{2,3})
local t6 = prior_box(500,2,{max=475,min=395},{2,3})
local t7 = prior_box(500,1,{max=555,min=475},{2,3})

--flat

--concat

return {t1, t2 ,t3, t4, t5 ,t6, t7}
end



function matching_gt_matrix(gt) -- xmin, ymin, xmax, ymax

local real_size = total_box()

local matched ={}

for iter = 1, #real_size do
local matching_for_tensor = torch.gt(jaccard_matrix(real_size[iter],gt),0.5)
table.insert(matched,matching(real_size[iter],matching_for_tensor))
end

return matched
end
