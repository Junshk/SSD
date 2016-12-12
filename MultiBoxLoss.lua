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
  
  target[1] = target[1]:cuda()
  target[2] = target[2]:cuda()
  input[1] = input[1]:cuda()
  input[2] = input[2]:cuda()

 
  local target1 = target[1]:clone()
  local target2 = target[2]:clone()
  local input1 = input[1]:clone()
  local input2 = input[2]:clone()
  
  assert(torch.sum(torch.eq(target2,math.huge))==0)

  local element = torch.numel(target1)
  local negative_mask = torch.eq(target1,21):byte()
  local positive_num = element-torch.sum(negative_mask)
  local negative_num = math.min(element-positive_num,positive_num*3)
  local discard_negative_num = (element)-positive_num-negative_num
  local discard_mask, bd_conf
  local match_mask

  local logsoftmax_score = (input1)

  local softmax_score = Softmax:forward(torch.CudaTensor(input1:size()):copy(input1):transpose(2,3):transpose(1,2)):float()--torch.exp(logsoftmax_score):float()
  local softmax_result,ix_ = torch.max(softmax_score[{{1,20}}],1)
  --assert(softmax_result[{1,1,1}])
  --print(softmax_result:size(),input1:size())
  
  local sum_ex = torch.sum(softmax_score,1)[{1,1,1}]
  --print(sum_ex)
  assert(math.abs(sum_ex - 1)<1e-2,sum_ex) 
  local t2 = sys.clock()
  softmax_result = softmax_result:view(batch,default_boxes)
  assert(torch.sum(torch.ne(input1,input1) )==0 , 'nan in  input1')
-------------------------------------------------------------------------
  
  bd_conf,ix_bd = torch.topk(torch.add(softmax_result,(1-negative_mask):float()*2):view(-1),discard_negative_num,1,false,true)
 --print('neg max',torch.max(softmax_result[negative_mask]:view(-1)),'min',torch.min(softmax_result[negative_mask]:view(-1)))
 --print('softmax max',torch.max(softmax_score),'min',torch.min(softmax_score))
  

 --print(torch.max(bd_conf),'max bd_conf')
 --assert(nil)
  discard_mask = torch.ByteTensor(batch, default_boxes):fill(0)
  match_mask = torch.ByteTensor(batch,default_boxes):fill(0)
  excp21_match_mask = torch.ByteTensor(batch, default_boxes):fill(0)

  for i_bd = 1, ix_bd:numel() do
    discard_mask:view(-1)[ix_bd[i_bd]] = 1   

  end
  bd_conf = bd_conf[bd_conf:numel()]
  assert(bd_conf<=1)
  local discard_iter = 1
  for b_iter = 1, batch do
    for d_iter = 1, default_boxes do
      local softmax_value = softmax_result[{b_iter,d_iter}]
      local discard_mask_value = discard_mask[{b_iter,d_iter}]--:squeeze()
      assert(type(softmax_value)=='number' and type(discard_mask_value)=='number',
      type(softmax_value)..' '.. type(discard_mask_value))
            
      if --softmax_value  <= bd_conf and 
        discard_mask_value == 1 --and 
       -- discard_iter <= discard_negative_num
        then
        --discard_mask[{b_iter,d_iter}] = 1
        --discard_iter = discard_iter + 1
       
      else  
        local __,conf_class = torch.max(softmax_score[{{},b_iter,d_iter}],1)
        local __21 , conf_class_ex21 = torch.max(softmax_score[{{1,20},b_iter,d_iter}],1)
        local target_sample = target1[{b_iter,d_iter}]:squeeze()
        assert(type(target_sample)=='number')
        --print(target,conf_class:squeeze())
        if  target_sample ==  conf_class:squeeze() then
          match_mask[{b_iter,d_iter}] = 1
          if target_sample == conf_class_ex21:squeeze() then
                  excp21_match_mask[{b_iter,d_iter}] = 1
          end
        end
      end  
    end
  end

 -- print('d iter',discard_iter)
  local match_num = torch.sum(match_mask)
  local p_match_mask = torch.cmul(match_mask,1-negative_mask)
  local p_match_num = torch.sum(p_match_mask)
  local n_match_num = match_num -p_match_num
  local excp21_p_match_num = torch.sum(excp21_match_mask)
  assert(torch.sum(discard_mask)==discard_negative_num,'discard_num')
  assert(p_match_num <= positive_num,'p_match<=p_num')
--------------------------------
  local dl_dx_loc 
  local dl_dx_conf 

  local loss_conf ,loss_loc = 0,0
  
-- forward

  local Grad = {}
  Grad[1] = torch.CudaTensor(input[1]:size()):fill(0)
  Grad[2] = torch.CudaTensor(input[2]:size()):fill(0)
  --print(input[1]:size(),target[1]:size())

  assert(negative_mask:dim() == 3,'negmask '..negative_mask:dim())
  --assert(discard_mask:dim() == 3,'discmask '..discard_mask:dim())

  local l1 = nn.SmoothL1Criterion():cuda() ; l1.sizeAverage = false
  local CE = nn.CrossEntropyCriterion():cuda() ; CE.nll.sizeAverage = false
  local err = 0

  for i_batch = 1, batch do
    for i_box = 1, default_boxes do
     
     assert(type(negative_mask[{1,1}]:squeeze()) =='number') 
      if negative_mask[{i_batch,i_box}]:squeeze() == 0 then
        loss_loc = loss_loc + l1:forward(input2[{i_batch,i_box}],target2[{i_batch,i_box}])
        
        --Grad[2][{i_batch,i_box}] = 
        Grad[2][{i_batch,i_box}]:copy( l1:backward(input2[{i_batch,i_box}], target2[{i_batch,i_box}]))
      
      end
      assert(type(discard_mask[{1,1}])=='number')
      if discard_mask[{i_batch,i_box}] ==0 then
		if target1[{i_batch,i_box}] == 21 then print(21) end
        loss_conf = loss_conf + CE:forward(input1[{i_batch,i_box}],target1[{i_batch,i_box}]:squeeze())
        Grad[1][{i_batch,i_box}]:copy( CE:backward(input1[{i_batch,i_box}],target1[{i_batch,i_box}]):squeeze())
      end
    end
  end
   
  local err = loss_conf + loss_loc
  --discard conf
--resize

  local t4 = sys.clock()

  local n = (positive_num)--+negative_num) ;  

  if n ==0 then   Grad[1]:fill(0);Grad[2]:fill(0) ; err = 0  end--dl_dx:fill(0) ; n = 0 end
 -- if n~= 0 then Grad[1]:div(n); Grad[2]:div(n) ; err = err/n end
  
  --local accuracy = match_num*100
 -- local _,max_exc_21 = torch.max(input[1][{{},{1,20}}],2)


  local accuracy = p_match_num*100
  local accuracy_n = positive_num
  print('bdconf',bd_conf)
  print('loss',loss_conf, loss_loc)
  print('match',p_match_num,n_match_num,match_num,'ex21', excp21_p_match_num)
--  print('except 21',torch.sum(p_match_exc21_mask))
  print('np',positive_num,negative_num,discard_negative_num)
  --print('acc',accuracy,accuracy_n)
  print('mask', torch.sum(negative_mask),torch.sum(discard_mask))
  print(' ')
  local t5 =sys.clock()
--  if loss_conf+loss_loc>1e+5*(n+2) then assert(nil,'huge loss') end 
  assert(match_num<=positive_num+negative_num, 'wrong match_num '..match_num..' '..positive_num.. ' '..negative_num..' '..discard_negative_num)
  assert(positive_num+negative_num+discard_negative_num == element)
  
  collectgarbage();
  
   return err, Grad,n , accuracy, accuracy_n
  
end





