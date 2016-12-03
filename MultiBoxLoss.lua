require 'cunn'
require 'cudnn'

torch.setdefaulttensortype('torch.FloatTensor')
local softmax = nn.LogSoftMax():cuda()
local L1loss = nn.SmoothL1Criterion():cuda()
L1loss.sizeAverage = false
local weight = torch.Tensor(21):fill(1);
if weighted21 ==true then weight[21]=1/3 end
local nll = nn.ClassNLLCriterion(weight):cuda()
nll.sizeAverage =false
---------------------------------------
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
  local loss = 0
  local batch ; if input:dim()==3 then batch= input:size(1) else batch=1 end

  local t1 = sys.clock()
  local input1 , input2 = bat2wo(input[{{},{1,21}}]),bat2wo(input[{{},{22,25}}])
  local target1, target2 =bat2wo(target[{{},{5}}]):cuda(), bat2wo(target[{{},{1,4}}]):cuda()
 


  local element = torch.numel(target1)
  local negative_mask = torch.eq(target1,21):byte()
  local positive_num = element-torch.sum(negative_mask)
  local negative_num = math.min(element-positive_num,positive_num*3)
  local discard_negative_num = (element)-positive_num-negative_num
  local discard_mask, bd_conf
  local nomatch_mask

  negative_mask = bat2wo(negative_mask)
-- print(input1:size()) 
  local logsoftmax_score = softmax:forward(input1:cuda()):float()
  local softmax_score = torch.exp(logsoftmax_score)
  local softmax_result,ix_ = torch.max(softmax_score[{{},{1,20}}],2)
  input1:float()

  local t2 = sys.clock()

  assert(torch.sum(torch.ne(input1,input1) )==0 , 'nan in  input1')

-------------------------------------------------------------------------
  local score, k = torch.topk(torch.csub(softmax_result,negative_mask:float()*2),discard_negative_num,1,false,true)

--print('score bg',score[{discard_negative_num}]:squeeze()+2)
  discard_mask = torch.ByteTensor(softmax_result:size()):fill(0)
if k:numel() ~= discard_negative_num then assert(nil) end
for iter = 1, discard_negative_num do
 
 discard_mask[ k[iter]:squeeze()] =1
end
discard_mask:cmul(negative_mask)
 if torch.sum(discard_mask) ~= discard_negative_num then 
   assert(nil,discard_negative_num..' '..torch.sum(discard_mask)) end 
--discard, remove loc of 21(bg) 
  
  local _, input1_max = torch.max(softmax_score:float(),2)

  local match_mask = (torch.eq(input1_max,target1:long()))

  p_match_mask = torch.cmul(match_mask,1-negative_mask)
  match_mask:cmul(1-discard_mask)
  local match_num = torch.sum(match_mask)
  local p_match_num = torch.sum(p_match_mask)

  local t3= sys.clock()

local n_match_mask = torch.cmul(match_mask,negative_mask)
local n_match_num = torch.sum(n_match_mask)
--------------------------------
  local dl_dx_loc 
  local dl_dx_conf 

  local loss_conf ,loss_loc = 0,0
  
 
-- forward
  local conf_mask = torch.ByteTensor(input1:size()):fill(1) -- no use
  for i = 1, conf_mask:size(1) do
    if target1[i]:squeeze() < 21 then 
      conf_mask[{{i},target1[{{i}}]}] = 0
    elseif discard_mask[i]:squeeze() == 0 then
      conf_mask[{{i},{21}}] = 0
    end
  end  


--  loss_conf = CrossEntropy:forward((input1):cuda(),(target1):cuda())
  logsoftmax_score[discard_mask:expand(logsoftmax_score:size())] = 0
  loss_conf = nll:forward((logsoftmax_score):cuda(),(target1:squeeze()):cuda())
  dl_dx_conf_ = nll:backward(logsoftmax_score:cuda(),target1:squeeze():cuda()) 
  dl_dx_conf_[discard_mask:expand(dl_dx_conf_:size())] = 0 
  dl_dx_conf = softmax:backward(input1:cuda(),dl_dx_conf_:cuda()):float()
  dl_dx_conf[discard_mask:expand(dl_dx_conf_:size())] = 0 

   target2[negative_mask:expand(target2:size())] =0
  input2[negative_mask:expand(input2:size())] = 0
  loss_loc = L1loss:forward((input2):cuda(),(target2):cuda())*lambda
  dl_dx_loc =  L1loss:backward((input2):cuda(),(target2):cuda()):float()*lambda
  dl_dx_loc[negative_mask:expand(dl_dx_loc:size())]= 0
 
--discard conf
  --dl_dx_loc:cmul((1-negative_mask):expand(dl_dx_loc:size()):float())

--resize

  dl_dx_loc = wo2bat(dl_dx_loc,batch)
  dl_dx_conf = wo2bat(dl_dx_conf,batch)
 
  local t4 = sys.clock()

  local dl_dx = torch.cat({dl_dx_conf,dl_dx_loc},2)
  local n = (positive_num)--+negative_num) ;  
  
  if n ==0 then loss_conf =0; loss_loc =0; dl_dx:fill(0) ; n = 0 end
  --local accuracy = match_num*100
  local _,max_exc_21 = torch.max(input1[{{},{1,20}}],2)
  local p_match_exc21_mask = torch.eq(max_exc_21:long(),target1:long())
  local accuracy = torch.sum(torch.cmul(p_match_exc21_mask:long(),torch.gt(_,0.01):long()))*100
  local accuracy_n = torch.sum(torch.gt(_,0.01):long():cmul((1-negative_mask):long()))
  
  print('loss',loss_conf, loss_loc)
  print('match',p_match_num,n_match_num,match_num)
  print('except 21',torch.sum(p_match_exc21_mask))
  print('np',positive_num,negative_num,discard_negative_num)
  print(' ')
  local t5 =sys.clock()
--  if loss_conf+loss_loc>1e+5*(n+2) then assert(nil,'huge loss') end 
  assert(match_num<=positive_num+negative_num, 'wrong match_num '..match_num..' '..positive_num.. ' '..negative_num..' '..discard_negative_num)
  assert(positive_num+negative_num+discard_negative_num == element)
  
--  print(t1,t2,t3,t4,t5)
  --local negative_match_mask = torch.eq(input1_max,21):cmul(negative_mask)
  --local negative_match_num = torch.sum(negative_match_mask)
  collectgarbage();
  
   return (loss_conf+loss_loc), dl_dx,n , accuracy, accuracy_n
  
end





