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
function jaccard_matrix(tensor,gt) 
if tensor:dim() ==4 then
gt:resize(4,1,1,1)
end

local expandGt = gt:expandAs(tensor) -- 4 by ~

local xdelta = torch.cmin(tensor[{{3}}],expandGt[{{3}}])-
torch.cmax(tensor[{{1}}],expandGt[{{1}}])
local ydelta = torch.cmin(tensor[{{4}}],expandGt[{{4}}])-
torch.cmax(tensor[{{2}}],expandGt[{{2}}])
xdelta:cmax(0)
ydelta:cmax(0)
local I = torch.cmul(xdelta,ydelta)

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

-- In addition to above, this code is adjusted version of fast nms for SSD. 
-- The authors of SSD use cuda to solve nms more faster 

function nms(boxes_mm, overlap, scores,image_size) -- adjusted
--print(torch.max(scores))
  boxes_mm = boxes_mm:float(); 
 
  local pick = torch.LongTensor()

  if boxes_mm:numel() == 0 then
    return pick
  end

  boxes_mm = boxes_mm:view(-1,4)

  local box_num = boxes_mm:size(1)

  local a1 = sys.clock()
  
  boxes_mm = boxes_mm:cuda()
--  local w_2 = boxes_mm[{{},{1}}]:div(2)
--  local h_2 = boxes_mm[{{},{2}}]:div(2)
  
  boxes_mm[{{},{3}}]:add(1/2,boxes_mm[{{},{1}}])
  boxes_mm[{{}, {4}}]:add(1/2,boxes_mm[{{},{2}}])--h_2) 
--  boxes_mm[{{},{1,2}}]:copy(boxes_mm[{{},{3,4}}])
  
--  boxes_mm[{{},{1}}]:csub(w_2)
--  boxes_mm[{{}, {2}}]:csub( h_2)
  boxes_mm[{{},{1}}]:add(boxes_mm[{{},{3}}],-1,boxes_mm[{{},{1}}])
  boxes_mm[{{},{2}}]:add(boxes_mm[{{},{4}}],-1,boxes_mm[{{},{2}}])
 
  boxes_mm:clamp(0,1)

  local S, h = torch.csub(boxes_mm[{{},{3}}],boxes_mm[{{},{1}}]), torch.csub(boxes_mm[{{},{4}}],boxes_mm[{{},{2}}])
  S:cmul(h) -- reuse
  local xmax = boxes_mm[{{},{3}}]
  xmax= torch.cmin(xmax:expand(box_num,box_num),xmax:t():expand(box_num,box_num))

  local xmin = boxes_mm[{{},{1}}]
  xmax:csub(torch.cmax(xmin:expand(box_num,box_num),xmin:t():expand(box_num,box_num)))-- = torch.cmax(xmin:expand(box_num,box_num),xmin:t():expand(box_num,box_num))


  
   local a2 = sys.clock()

  local ymax = boxes_mm[{{},{4}}]
  ymax = torch.cmin(ymax:expand(box_num,box_num),ymax:t():expand(box_num,box_num))
  local ymin = boxes_mm[{{},{2}}]
  ymax:csub(torch.cmax(ymin:expand(box_num,box_num),ymin:t():expand(box_num,box_num)))


   local a3 =sys.clock()
   
 
  local inter = torch.cmul(ymax,xmax)

  local IOU = torch.cdiv(inter,S:expand(box_num,box_num)+S:t():expand(box_num,box_num)-inter)


  
 
  scores = scores:cuda()
local   v, Order = scores:sort(1)
  Order=Order:long()


local a4= sys.clock()




  pick:resize(box_num):zero()
  count = 1

while true do 
 if Order == nil then print('order nil') break end
 if Order:numel() == 0 then break end
    
    local last = Order:size(1)
    local i = Order[last]
    
    
  --  print(count,i)
    pick[count] = i
    count = count + 1
    
    if last == 1 then
      break
    end
    

    Order = Order[{{1, last-1}}] -- remove picked element from view
    local partial_IoU = IOU[i]:float():squeeze():index(1,Order)

    Order = Order[partial_IoU:le(overlap)] -- keep only elements with a IoU < overlap
  --  print(pick:size())
end

  -- reduce size to actual count
  local a5 = sys.clock()
  local top = 200 
  count = math.min(count,top)

  pick = pick[{{1, count-1}}]
--print(pick:size()) 
  boxes_mm[{{},{1}}]:mul(image_size[3])
  boxes_mm[{{},{2}}]:mul(image_size[2])
  boxes_mm[{{},{3}}]:mul(image_size[3])
  boxes_mm[{{},{4}}]:mul(image_size[2])

  -- remove wrong box
  S = S:float()
  pick = pick[torch.ne(S:index(1,pick),0)]
--  print(pick:size())
  local a6 =sys.clock()
--  print(a6-a5,a5-a4,a4-a3,a3-a2,a2-a1)
  
  boxes_mm = boxes_mm:float()
  scores = scores:float()
--print(pick)
   boxes_mm = boxes_mm:index(1,pick)
   scores = scores:index(1,pick)
--   print(torch.max(scores))
   return boxes_mm, scores
end
