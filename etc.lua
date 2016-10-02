require 'image'



function jaccard(anno1,anno2)
-- format : cx cy w h
local I,U
if type(anno) == 'table' then
I = (math.min(anno1.cx+anno1.w/2,anno2.cx+anno2.w/2)-math.max(anno1.cx-anno1.w/2,anno2.cx-anno2.w/2))
*(math.min(anno1.cy+anno2.h/2,anno2.cy+anno2.h/2)-math.max(anno1.cy-anno1.h/2,anno2.cy-anno2.h/2))
U = anno1.w*anno1.h+anno2.w*anno2.h - I

return I/U
else -- tensor n by 4 


I = (torch.min(anno1[{{},{1}}]+anno1[{{},{3}}]/2,anno2[{{},{1}}]+anno2[{{},{3}}])-torch.max(anno1[{{},{1}}]-anno1[{{},{3}}]/2,anno2[{{},{1}}]-anno2[{{},{3}}])):cmul(torch.min(anno1[{{},{2}}]+anno1[{{},{4}}]/2,anno2[{{},{2}}]+anno2[{{},{4}}])-torch.max(anno1[{{},{2}}]-anno1[{{},{4}}]/2,anno2[{{},{2}}]-anno2[{{},{4}}]))

U = torch.cmul(anno1[{{},{3}}],anno2[{{},{4}}])- I
return I:cdiv(U)
end

end


function jaccard_matrix(tensor,gt) 
gt:resize(4,1,1,1)


local expandGt = gt:expandAs(tensor) -- 4 by ~

local xdelta = torch.min(tensor[{{3}}],expandGt[{{3}}])-
torch.max(tensor[{{1}}],expandGt[{{1}}])
local ydelta = torch.max(tensor[{{4}}],expandGt[{{4}}])-
torch.max(tensor[{{2}}],expandGt[{{2}}])

local I = xdelta:cmul(ydelta)


local U = (tensor[{{3}}]-tensor[{{1}}]):cmul(tensor[{{4}}]-tensor[{{2}}])
+(gt[{{3}}]-gt[{{1}}])*(gt[{{4}}]-gt[{{2}}]) - I



return I:cdiv(U)

end





--[[
function augmentation(img,anno) 
local random = math.random(1,3)



if random ==1 then
--original

elseif random ==2 then
--jaccard 0.1 0.3 0.5 0.7 0.9


elseif random==3 then
-- random

end


-- if hflip
if math.random(1,2) ==2 then
img = image.hflip(img)
end

return img
end
]]--

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
function nms(boxes, overlap, scores)
   local pick = torch.LongTensor()

  if boxes:numel() == 0 then
    return pick
  end

  local x1 = boxes[{{}, 1}]
  local y1 = boxes[{{}, 2}]
  local x2 = boxes[{{}, 3}]
  local y2 = boxes[{{}, 4}]
    
  local area = torch.cmul(x2 - x1 + 1, y2 - y1 + 1)
  
  if type(scores) == 'number' then
    scores = boxes[{{}, scores}]
  elseif scores == 'area' then
    scores = area
  else
    scores = y2   -- use max_y
  end
  
  local v, I = scores:sort(1)

  pick:resize(area:size()):zero()
  local count = 1
  
  local xx1 = boxes.new()
  local yy1 = boxes.new()
  local xx2 = boxes.new()
  local yy2 = boxes.new()

  local w = boxes.new()
  local h = boxes.new()

  while I:numel() > 0 do 
    local last = I:size(1)
    local i = I[last]
    
    pick[count] = i
    count = count + 1
    
    if last == 1 then
      break
    end
    
    I = I[{{1, last-1}}] -- remove picked element from view
    
    -- load values 
    xx1:index(x1, 1, I)
    yy1:index(y1, 1, I)
    xx2:index(x2, 1, I)
    yy2:index(y2, 1, I)
    
    -- compute intersection area
    xx1:cmax(x1[i])
    yy1:cmax(y1[i])
    xx2:cmin(x2[i])
    yy2:cmin(y2[i])
    
    w:resizeAs(xx2)
    h:resizeAs(yy2)
    torch.add(w, xx2, -1, xx1):add(1):cmax(0)
    torch.add(h, yy2, -1, yy1):add(1):cmax(0)
    
    -- reuse existing tensors
    local inter = w:cmul(h)
    local IoU = h
    
    -- IoU := i / (area(a) + area(b) - i)
    xx1:index(area, 1, I) -- load remaining areas into xx1
    torch.cdiv(IoU, inter, xx1 + area[i] - inter) -- store result in iou
    
    I = I[IoU:le(overlap)] -- keep only elements with a IoU < overlap 
  end

  -- reduce size to actual count
  pick = pick[{{1, count-1}}]
  return pick
end
