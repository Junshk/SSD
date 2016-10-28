require 'nn'
require 'cunn'
require 'cudnn'

-- add channel shared option to nn.mul:
-- it is for SSD

local Mul_modified , parent = torch.class('nn.Mul_modified','nn.Module')


function Mul_modified:__init(channel,init_scale) -- descript channel num manual ;;
  parent.__init(self)
  self.weight = torch.Tensor(1,channel,1,1)
  self.gradWeight = torch.Tensor(1,channel,1,1)
  self.ch = channel or 3 
  self:reset(init_scale)
end


function Mul_modified:reset(init_scale)

--  if stdv then
--          stdv = stdv*math.sqrt(3)
--          self.weight:uniform(-stdv,stdv);
--  else
         -- stdv = 1./math.sqrt(self.weight:size(1))
         self.weight:fill(init_scale)
--  end

end

function Mul_modified:updateOutput(input)

  if input:dim() == 3 then input = input:view(1,input:size(1),input:size(2),input:size(3))  end
--          elseif input:dim() == 4 then self.weight=self.weight:view(1,self.ch,1,1) end

  self.output:resizeAs(input):copy(input)

  


self.output:cmul(self.weight:expandAs(input))--self.expand)
  return self.output
end

function Mul_modified:updateGradInput(input ,gradOutput)



  self.gradInput:resizeAs(input):zero()


 self.gradInput:addcmul(1,self.weight:expandAs(gradOutput),gradOutput)

return self.gradInput

end

function Mul_modified:accGradParameters(input, gradOutput, scale)--
  scale = scale or 1
    if input:dim() == 4 then self.gradWeight =self.gradWeight:view(1,self.ch,1,1) elseif input:dim() == 3 then self.gradWeight = self.gradWeight:view(self.ch,1,1) end

if input:dim() ==4 then  self.gradWeight = self.gradWeight +scale*input:cmul(gradOutput):mean(input:dim()):mean(input:dim()-1):mean(1) end
  
  
end
--[[
function Mul_modified:clearState()
nn.utils.clear(self,{
        
        
        
        })


end
]]--
