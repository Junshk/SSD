require 'cunn'
require 'cudnn'

torch.setdefaulttensortype('torch.FloatTensor')
local softmax = nn.LogSoftMax():cuda()
softmax:evaluate()
----------------------------------------
function bat2wo(input)

  if input:dim() ==2 then return input end
  local size = input:size()
  local reshaped = input:transpose(2,3):reshape(1,size[1]*size[3],size[2]):squeeze(1)

  return reshaped
end
----------------------------------------
function wo2bat(input,batch)

  local size = input:size()
  assert(input:dim()==2,'wo2bat size error ')
  local class = size[2]
  local n = size[1]/batch
          
  local reshaped = input:reshape(batch,n,class):transpose(2,3)

  return reshaped
end
-----------------------------------------
function MultiBoxLoss(input,target,lambda)  -- target1 : class 1 by pyramid, bd 4 by pyramid

--  local alpha = 1
  local loss = 0
  local batch ; if input:dim()==3 then batch= input:size(1) else batch=1 end

  local element = torch.numel(target[1])
  local negative_mask = torch.eq(target[1],21)
  local positive_num = element-torch.sum(negative_mask)
  local negative_num = math.min(element-positive_num,positive_num*3)
  local discard_negative_num = (element)-positive_num-negative_num
  local discard_mask, bd_conf
  local nomatch_mask

  local input1 , input2 = bat2wo(input[{{},{1,21}}]),bat2wo(input[{{},{22,25}}])
  local target1, target2 =bat2wo(target[1]), bat2wo(target[2])
  negative_mask = bat2wo(negative_mask)
  
  local softmax_score = softmax:forward(input1:cuda()):float():exp()
  local softmax_result = softmax_score[{{},{21}}]
  input1:float()



-------------------------------------------------------------------------
  if discard_negative_num~=0 then
    local score, k = torch.topk(softmax_result[negative_mask],discard_negative_num,1,false,true)
    bd_conf = score[{{discard_negative_num}}]:squeeze()
    else bd_conf =0 end

  discard_mask = torch.le(softmax_result,bd_conf):cmul(negative_mask)

--discard, remove loc of 21(bg) 
  target2[negative_mask:expandAs(target2)]= input2[negative_mask:expandAs(input2)] 
  
  local _, input1_max = torch.max(input1,2)

 target1[discard_mask:expandAs(target1)] = input1_max[discard_mask:expandAs(target1)]:float()
 -- nomatch_mask = torch.ne(input1_max, target1:long()) 
  --target2[nomatch_mask:expandAs(target2)] = input2[nomatch_mask:expandAs(input2)]:float()

--  assert(nil)
  local match_num = torch.sum(torch.eq(input1_max[1-discard_mask],target1[1-discard_mask]:long()))
--print('mask',torch.sum(negative_mask),'>',torch.sum(discard_mask))
--print(torch.sum(discard_mask),'=',element-negative_num-positive_num)








--------------------------------
  local dl_dx_loc = torch.Tensor(target2)
  local dl_dx_conf 

  local L1loss = nn.SmoothL1Criterion():cuda()
  L1loss.sizeAverage = false
  local CrossEntropy = nn.CrossEntropyCriterion():cuda()
  CrossEntropy.nll.sizeAverage = false
  local loss_conf ,loss_loc = 0,0


--  input:float()
-- forward

  loss_conf = CrossEntropy:forward((input1):cuda(),(target1):cuda())
  loss_loc = L1loss:forward((input2):cuda(),(target2):cuda())*lambda
  dl_dx_loc =  L1loss:backward((input2):cuda(),(target2):cuda()):float()*lambda

  L1loss = nil;
--  input2 = nil ;
  target2:float();
  input:float()
    dl_dx_conf = CrossEntropy:backward((input1):cuda(),target1:cuda()):float()
--  input1 = nil;
  input1:float()
  target1:float();
  CrossEntropy = nil

--discard conf
 
--resize

  dl_dx_loc = wo2bat(dl_dx_loc,batch)
  dl_dx_conf = wo2bat(dl_dx_conf,batch)
--   if loss_loc == math.huge then
--print(target[2][{{1,4},{},{1,4}}],--input[{{1,4},{22,25},{1,4}}],
--torch.sum(torch.eq(target[2],math.huge)))
--    end
  
  local dl_dx = torch.cat({dl_dx_conf,dl_dx_loc},2)
  local n = math.max(positive_num+negative_num,1) ; --if n ==0 then n =1; print('n ==0')end 
  local accuracy = match_num*100
  collectgarbage();
  assert(match_num<=positive_num+negative_num, 'wrong match_num '..match_num..' '..positive_num.. ' '..negative_num)
  return (loss_conf+loss_loc), dl_dx,n, accuracy

end






