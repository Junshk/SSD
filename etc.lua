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

function jaccard(anno1,anno2)

local xmin, xmax = torch.cmax(anno1[{{1}}],anno2[{{1}}]), torch.cmin(anno1[{{3}}],anno2[{{3}}])
local ymin, ymax = torch.cmax(anno1[{{2}}], anno2[{{2}}]), torch.cmin(anno1[{{4}}],anno2[{{4}}])

local I = torch.cmax(xmax-xmin,0):cmul( torch.cmax(ymax-ymin,0))

local U = (anno1[{{3}}]-anno1[{{1}}]):cmul(anno1[{{4}}]-anno1[{{2}}])  + (anno2[{{3}}]-anno2[{{1}}]):cmul(anno2[{{4}}]-anno2[{{2}}]) - I 

return I:cdiv(U)
end

--end

-----------------------------------------------------------
function jaccard_matrix(tensor,gt) 
if tensor:dim() ==4 then
gt:resize(4,1,1,1)
end

local expandGt = gt:expandAs(tensor) -- 4 by ~

local xdelta = torch.cmin(tensor[{{3}}],expandGt[{{3}}])-
torch.cmax(tensor[{{1}}],expandGt[{{1}}])
local ydelta = torch.cmax(tensor[{{4}}],expandGt[{{4}}])-
torch.cmax(tensor[{{2}}],expandGt[{{2}}])

local I = xdelta:cmul(ydelta)

local tensor_size = tensor:size(); tensor_size[1]=1
local tensor_element = torch.numel(tensor[{{1}}])
local t_t = (tensor[{{3}}]-tensor[{{1}}]):reshape(tensor_element)
t_t:cmul(tensor[{{4}}]-tensor[{{2}}]:reshape(tensor_element))

gt=gt:squeeze()
local U =t_t:reshape(tensor_size)+(gt[{{3}}]-gt[{{1}}])*(gt[{{4}}]-gt[{{2}}]) - I



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
function nms(boxes, overlap, scores,image_size) -- adjusted
  boxes = boxes:float(); scores = scores:float()
--  local score_upper = torch.gt(scores:view(-1,1),0.01)
--  scores = scores[score_upper]


--  score_upper = score_upper:expandAs(boxes)


--  boxes = boxes[score_upper]
  --  boxes = torch.reshape(boxes,boxes:numel(),4)
  

  
  local pick = torch.LongTensor()

  if boxes:numel() == 0 then
    return pick
  end

  boxes = boxes:view(-1,4)

  local box_num = boxes:size(1)
--  local all_pairs_iou = torch.Tensor(box_num,box_num)

  local a1 = sys.clock()
  local boxes_mm = torch.Tensor(boxes)

  boxes_mm[{{},{1}}] = boxes[{{},{ 3}}] - boxes[{{},{1}}]/2
  boxes_mm[{{}, {2}}] = boxes[{{},{ 4}}] - boxes[{{},{2}}]/2
  boxes_mm[{{},{3}}] = boxes[{{},{ 3}}] + boxes[{{},{1}}]/2
  boxes_mm[{{}, {4}}] = boxes[{{},{ 4}}] + boxes[{{},{2}}]/2
  
  boxes_mm:cuda()
  boxes_mm:clamp(0,1)

  local w, h = boxes_mm[{{},{3}}]-boxes_mm[{{},{1}}], boxes_mm[{{},{4}}]-boxes_mm[{{},{2}}]

  local xmax = boxes_mm[{{},{3}}]:expand(box_num,box_num)
  xmax = torch.cmin(xmax,xmax:t())
  local xmin = boxes_mm[{{},{1}}]:expand(box_num,box_num)
  xmin = torch.cmax(xmin,xmin:t())
  local ymax = boxes_mm[{{},{4}}]:expand(box_num,box_num)
  ymax = torch.cmin(ymax,ymax:t())
  local ymin = boxes_mm[{{},{2}}]:expand(box_num,box_num)
  ymin = torch.cmax(ymin,ymin:t())


  local inter = torch.cmul(ymax-ymin,xmax-xmin)
--  ymax=nil;ymin=nil;xmax=nil;xmin=nil; collectgarbage();

  local w , h = boxes_mm[{{},{3}}] - boxes_mm[{{},{1}}], boxes_mm[{{},{4}}] - boxes_mm[{{},{2}}]
  
  local S = torch.cmul(w,h):float()
  local area = torch.cmul(w:expand(box_num,box_num),h:expand(box_num,box_num)):cmax(0)
--  inter[torch.eq(area,0)] = 1
--  area[torch.eq(area,0)] = 1

  local IOU = torch.cdiv(inter,area+area:t()-inter)
  IOU = IOU:float()
  
  
  local a2 = sys.clock()
  
  local v, Order = scores:cuda():sort(1)
  Order=Order:long()
  v = nil;





  pick:resize(box_num):zero()
  local count = 1
  --[[
  local xx1 = boxes.new()
  local yy1 = boxes.new()
  local xx2 = boxes.new()
  local yy2 = boxes.new()

  local ww = boxes.new()
  local hh = boxes.new()
]]--


  local a3= sys.clock()
 while Order:numel() > 0 do 
    local last = Order:size(1)
    local i = Order[last]
    
    

    pick[count] = i
    count = count + 1
    
    if last == 1 then
      break
    end
    

    Order = Order[{{1, last-1}}] -- remove picked element from view
    local partial_IoU = IOU[i]:squeeze():index(1,Order)

    Order = Order[partial_IoU:le(overlap)] -- keep only elements with a IoU < overlap 
--  Order = Order:cmul(partial_IoU)

--if torch.gt(Order,0):numel() ==0 then break;end
end

  -- reduce size to actual count
  local a4 = sys.clock()
  local top = 200 
  count = math.min(count,top)
  pick = pick[{{1, count-1}}]

  boxes_mm[{{},{1}}]:mul(image_size[3])
  boxes_mm[{{},{2}}]:mul(image_size[2])
  boxes_mm[{{},{3}}]:mul(image_size[3])
  boxes_mm[{{},{4}}]:mul(image_size[2])

  -- remove wrong box
  pick = pick[torch.ne(S:index(1,pick),0)]

  boxes_mm = boxes_mm:float()
  scores = scores:float()
   return boxes_mm:index(1,pick), scores:index(1,pick)
end
