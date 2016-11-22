require 'image'
require 'sys'
torch.setdefaulttensortype('torch.FloatTensor')
local Class ={'aeroplane','bicycle','bird','boat','bottle','bus','car',
           'cat','chair','cow','diningtable','dog','horse','motorbike',
           'person','pottedplant','sheep','sofa','train','tvmonitor'}


-------------------------------------------------------
function class2num(class)


for k,v  in pairs(Class) do
  
  if v == class then 
                   
          return k end
end        

assert(nil,'wrong class name')

end

function num2class(num ) return Class[num] end
-----------------------------------------------------------
--[[
function jaccard(anno1,anno2)

local xmin, xmax = torch.cmax(anno1[{{1}}],anno2[{{1}}]), torch.cmin(anno1[{{3}}],anno2[{{3}}])
local ymin, ymax = torch.cmax(anno1[{{2}}], anno2[{{2}}]), torch.cmin(anno1[{{4}}],anno2[{{4}}])

local I = torch.cmax(xmax-xmin,0):cmul( torch.cmax(ymax-ymin,0))

local U = (anno1[{{3}}]-anno1[{{1}}]):cmul(anno1[{{4}}]-anno1[{{2}}])  + (anno2[{{3}}]-anno2[{{1}}]):cmul(anno2[{{4}}]-anno2[{{2}}]) - I 

return I:cdiv(U)
end

--end
]]--
-----------------------------------------------------------
function jaccard_matrix(tensor,gt) --xmin ymin xmax ymax 
if tensor:dim() ==4 then
gt = gt:view(4,1,1,1)
--print('s',gt)
end

--print('gt',gt)

local expandGt = gt:expandAs(tensor) -- 4 by ~
--print(tensor,'tensor')
local xdelta = torch.cmin(tensor[{{3}}],expandGt[{{3}}])-
torch.cmax(tensor[{{1}}],expandGt[{{1}}])
local ydelta = torch.cmin(tensor[{{4}}],expandGt[{{4}}])-
torch.cmax(tensor[{{2}}],expandGt[{{2}}])
--print(xdelta,ydelta)
xdelta:cmax(0)
ydelta:cmax(0)
--print('af',xdelta, ydelta)
local I = torch.cmul(xdelta,ydelta)

local tensor_size = tensor:size(); tensor_size[1]=1
--local tensor_element = torch.numel(tensor[{{1}}])
local t_t = (tensor[{{3}}]-tensor[{{1}}])
t_t:cmul((tensor[{{4}}]-tensor[{{2}}]))

gt=gt:squeeze()
local U =t_t:reshape(tensor_size)+(gt[{{3}}]-gt[{{1}}])*(gt[{{4}}]-gt[{{2}}]) - I

assert(torch.sum(torch.lt(U,0))==0 , 'jaccard error #'..torch.sum(torch.le(U,0)))
assert(torch.sum(torch.lt(I,0))==0,'jaccard error #'..torch.sum(torch.le(I,0)))

return I:cdiv(U)

end




-- Non-maximum suppression (NMS)
--
-- Greedily skip boxes that are significantly overlapping a previously 
-- selected box.
--
-- Arguments
--   boxes     Bounding boxes as nx4 tensor, each row specifies the
--             vertices of one box { min_x, min_y, max_x, max_y }. 
--   overlap   Intersection-over-union (IoU) threshold for suppression,
--             all boxes with va alues higher than this threshold will 
--             be suppressed.
--   scores    (optional) Defines in which order boxes are processed.
--             Either the string 'area' or a tensor holding 
--             score-values. Boxes will be processed sorted descending
--             after this value.
--
-- Return value
--   Indices of boxes remaining after non-maximum suppression.

-- Original author: Francisco Massa: https://github.com/fmassa/object-detection.torch 
-- Based on matlab code by Pedro Felzenszwalb https://github.com/rbgirshick/voc-dpm/blob/master/test/nms.m
-- Minor changes by Andreas Köpf, 2015-09-17 

-- In addition to above, this code is adjusted version of fast nms for SSD. 
-- The authors of SSD use cuda to solve nms more faster 

function nms(boxes_mm, overlap, scores,image_size) -- adjusted

  local top = 200 
  
  ----- cuda calculate iou of all pair
  boxes_mm = boxes_mm:float(); 
 
  local pick = torch.LongTensor()

  if boxes_mm:numel() == 0 then
    return pick
  end

  boxes_mm = boxes_mm:view(-1,4)

  local box_num = boxes_mm:size(1)
  local box_size = boxes_mm:size()

  local a1 = sys.clock()
  
  boxes_mm = boxes_mm:cuda()
 
  boxes_mm[{{},{3}}]:add(1/2,boxes_mm[{{},{1}}])
  boxes_mm[{{}, {4}}]:add(1/2,boxes_mm[{{},{2}}])--h_2) 
  boxes_mm[{{},{1}}]:add(boxes_mm[{{},{3}}],-1,boxes_mm[{{},{1}}])
  boxes_mm[{{},{2}}]:add(boxes_mm[{{},{4}}],-1,boxes_mm[{{},{2}}])
 
  boxes_mm:clamp(0,1)
  boxes_mm[{{},{1}}]:mul(image_size[3])
  boxes_mm[{{},{2}}]:mul(image_size[2])
  boxes_mm[{{},{3}}]:mul(image_size[3])
  boxes_mm[{{},{4}}]:mul(image_size[2])
  boxes_mm:floor()

  local S, h = torch.csub(boxes_mm[{{},{3}}],boxes_mm[{{},{1}}]), torch.csub(boxes_mm[{{},{4}}],boxes_mm[{{},{2}}])
  S:cmul(h) -- reuse
    h =nil; 
  

 


local a4= sys.clock()

  scores = scores:cuda()
  local v, Order = torch.sort(scores,1)
  Order=Order:long()

  pick:resize(box_num):zero()
  count = 1

while true do 
   if Order:numel() == 0 then break end
    
    local last = Order:size(1)
    local i = Order[last]
    
    
    --print(count,i)
    pick[count] = i
    count = count + 1
    if count>top then break ; end 
    if last == 1 then
      break
    end
    

    Order = Order[{{1, last-1}}] -- remove picked element from view
    local xmax =  torch.cmin(boxes_mm[{{},{3}}],boxes_mm[{i,3}])
    local xmin = torch.cmax(boxes_mm[{{},{1}}],boxes_mm[{i,1}])
    local ymax = torch.cmin(boxes_mm[{{},{4}}],boxes_mm[{i,4}])
    local ymin = torch.cmax(boxes_mm[{{},{2}}],boxes_mm[{i,2}])
    ymax:csub(ymin):cmax(0)
    xmax:csub(xmin):cmax(0)

    local I = torch.cmul(ymax,xmax)


    local IOU =torch.cdiv(I,( S[i]:squeeze()+S-I+1e-10))
    local partial_IoU = IOU:index(1,Order):view(Order:size()):float()
    --print(partial_IoU:le(overlap):size())

    Order = Order[partial_IoU:le(overlap)] -- keep only elements with a IoU < overlap
  
    xmax =nil
    ymax =nil
    ymin =nil
    xmin =nil
    collectgarbage()
  end

  -- reduce size to actual count
  local a5 = sys.clock()

  count = math.min(count,top)

  pick = pick[{{1, count-1}}]
   -- remove wrong box
  S = S:float()
  pick = pick[torch.ne(S:index(1,pick),0)]

  S =nil;
  local a6 =sys.clock()
  
  if pick:numel() == 0 then return pick end

    
  boxes_mm = boxes_mm:float()
  scores = scores:float()

  boxes_mm = boxes_mm:index(1,pick)
  scores = scores:index(1,pick)
   
  
  local output =   torch.Tensor(boxes_mm:size(1),6)
  
  output[{{},{1,4}}] = boxes_mm
  output[{{},{5}}] = scores
   collectgarbage()

 return output--boxes_mm, scores

end
