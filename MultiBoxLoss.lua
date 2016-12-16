require 'cunn'
require 'cudnn'

torch.setdefaulttensortype('torch.FloatTensor')

--local criterion = require 'loss'
--criterion:cuda()
local Softmax = nn.SoftMax():cuda()
---------------------------------------

-----------------------------------------
function MultiBoxLoss(input,target,lambda)  -- target1 : class 1 by pyramid, bd 4 by pyramid
  local loss = 0
  --local batch ; if input:dim()==3 then batch= input:size(1) else batch=1 end
  local batch = input[1]:size(1) ; if input[1]:dim()~=3 then assert(nil, 'no batch_mode') end
  local default_boxes = input[1]:size(input[1]:dim()-1)
 
 --print('batch',batch,'dboxes',default_boxes)
  local t1 = sys.clock()
  

  input[1] = input[1]:cuda()
  input[2] = input[2]:cuda()

 
  local target1 = target[1]:clone()
  local target2 = target[2]:clone()
  local input1 = input[1]:clone()
  local input2 = input[2]:clone()
  
  assert(torch.sum(torch.eq(target2,math.huge))==0)

  local element = torch.numel(target1)
  local negative_mask = torch.eq(target1,21):byte()
  local positive_num = batch-torch.sum(negative_mask,1)
  local negative_num = torch.cmin(batch-positive_num,positive_num*3)
--  local discard_negative_num = (element)-positive_num-negative_num
--  local discard_mask, bd_conf
--  local match_mask
--
  local logsoftmax_score = (input1)

  local softmax_score = Softmax:forward(torch.CudaTensor(input1:size()):copy(input1):transpose(2,3):transpose(1,2)):float()--torch.exp(logsoftmax_score):float()
  local softmax_pos_result,ix_pos = torch.max(softmax_score[{{1,20}}],1)
  local softmax_result, ix_ = torch.max(softmax_score,1)
--[[
  local sum_ex = torch.sum(softmax_score,1)[{1,1,1}]
  --print(sum_ex)
  assert(math.abs(sum_ex - 1)<1e-2,sum_ex) ]]--
  softmax_pos_result = softmax_pos_result:view(batch,default_boxes)
  softmax_result = softmax_result:view(batch,default_boxes)
  assert(torch.sum(torch.ne(input1,input1) )==0 , 'nan in  input1')
-------------------------------------------------------------------------
  
--  local bd_conf, ix_bd = torch.topk(torch.add(softmax_result,(1-negative_mask):float()*2):view(-1),discard_negative_num,1,false,true)
 --print('neg max',torch.max(softmax_result[negative_mask]:view(-1)),'min',torch.min(softmax_result[negative_mask]:view(-1)))
 --print('softmax max',torch.max(softmax_score),'min',torch.min(softmax_score))
  

 --print(torch.max(bd_conf),'max bd_conf')
 
  local match_mask = torch.eq(ix_:float(),target1:view(batch,default_boxes))
  local excp21_match_mask = torch.eq(ix_pos:float(),target1:view(batch,default_boxes))
  local discard_mask = torch.ByteTensor(batch,default_boxes):fill(0)


  local loss_conf ,loss_loc = 0,0
  local l1 = nn.SmoothL1Criterion():cuda() ; l1.sizeAverage = false
  local CE = nn.CrossEntropyCriterion():cuda() ; CE.nll.sizeAverage = false
  local err = 0


  local Grad = {}
  Grad[1] = torch.CudaTensor(input[1]:size()):fill(0)
  Grad[2] = torch.CudaTensor(input[2]:size()):fill(0)

    for d_iter = 1, default_boxes do
      local pos_sample_num = positive_num:squeeze()[{d_iter}];assert(type(pos_sample_num)=='number')
      local neg_sample_num = math.min(pos_sample_num*3,batch-pos_sample_num)
      local sort_conf ,sort_idx = torch.sort(softmax_score[{21,{},d_iter}],false)--torch.sort(softmax_pos_result[{{},d_iter}],true)
      local pos_iter = 1
      local neg_iter = 1
      
      if pos_sample_num >0 then
      
      local sample_loc_loss, sample_conf_loss = 0, 0
      
      for i_sample = 1, batch do
        local id = sort_idx[i_sample] --batchsampleid
        local pos =1- negative_mask[{id,d_iter}]:squeeze(); assert(type(pos)=='number')
        if pos == 1 then
          sample_loc_loss = sample_loc_loss + l1:forward(input2[{id,d_iter}],target2[{id,d_iter}]:cuda())
          Grad[2][{id,d_iter}]:copy(l1:backward(input2[{id,d_iter}],target2[{id,d_iter}]:cuda()))
          sample_conf_loss = sample_conf_loss + CE:forward(input1[{id,d_iter}],target1[{id,d_iter}]:cuda())
          Grad[1][{id,d_iter}]:copy(CE:backward(input1[{id,d_iter}],target1[{id,d_iter}]:cuda()))
          pos_iter = pos_iter + 1
        elseif  neg_iter <= neg_sample_num or (ix_[{id,d_iter}]~=21 and i_sample ==1) then
          sample_conf_loss = sample_conf_loss + CE:forward(input1[{id,d_iter}],target1[{id,d_iter}]:cuda())
          Grad[1][{id,d_iter}]:copy(CE:backward(input1[{id,d_iter}],target1[{id,d_iter}]:cuda()))
          neg_iter = neg_iter + 1
        else
          discard_mask[{id,d_iter}] = 1
        end        
       -- if pos_iter > pos_sample_num and neg_iter >neg_sample_num then break; end        
      end        -- for end
      assert(pos_iter>pos_sample_num and neg_iter>neg_sample_num,pos_iter..' '..pos_sample_num..' '..neg_iter..' '..neg_sample_num)
    --[[
      if pos_sample_num ~=0 then 
        Grad[1][{{},d_iter}]:div(pos_sample_num)
        Grad[2][{{},d_iter}]:div(pos_sample_num)
        sample_loc_loss =sample_loc_loss / pos_sample_num
        sample_conf_loss = sample_conf_loss / pos_sample_num
      end
      ]]--
      loss_conf = loss_conf + sample_conf_loss
      loss_loc = loss_loc + sample_loc_loss
      else discard_mask[{{},d_iter}] = 1 end--if end
    end  --- d for end
    
   local err = loss_conf + loss_loc
 

 -- print('d iter',discard_iter)
  local match_num = torch.sum(match_mask)
  local hard_match_num = torch.sum(torch.cmul(match_mask,1-discard_mask))
  local p_match_mask = torch.cmul(match_mask,1-negative_mask)
  local p_match_num = torch.sum(p_match_mask)
  local n_match_num = hard_match_num -p_match_num
  local excp21_p_match_num = torch.sum(torch.cmul(excp21_match_mask,1-negative_mask))
 -- assert(torch.sum(discard_mask)==discard_negative_num,'discard_num')
 -- assert(p_match_num <= positive_num,'p_match<=p_num')
--------------------------------
  local n = torch.sum(positive_num)
 
  local accuracy = p_match_num*100
  local accuracy_n =torch.sum( positive_num)
--  print('bdconf',bd_conf)
  print('loss',loss_conf, loss_loc)
  print('match',p_match_num,n_match_num,hard_match_num,match_num,'ex21', excp21_p_match_num)
--  print('except 21',torch.sum(p_match_exc21_mask))
  print('np',torch.sum(positive_num),torch.sum(negative_num))--,discard_negative_num)
  --print('acc',accuracy,accuracy_n)
  --print('mask', torch.sum(negative_mask),torch.sum(discard_mask))
  print(' ')
  local t5 =sys.clock()
--  if loss_conf+loss_loc>1e+5*(n+2) then assert(nil,'huge loss') end 
 -- assert(match_num<=positive_num+negative_num, 'wrong match_num '..match_num..' '..positive_num.. ' '..negative_num..' '..discard_negative_num)
--  assert(positive_num+negative_num+discard_negative_num == element)
  
  collectgarbage();
  
   return err, Grad,n , accuracy, accuracy_n
  
end





