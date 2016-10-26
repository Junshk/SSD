
require 'nn'
require 'cunn'
require 'cudnn'
-- I refer to torch nn Normalize.lua code, this code different little bit from that
local ChannelNormalization, parent = torch.class('nn.ChannelNormalization','nn.Module')


function ChannelNormalization:__init(p, eps)
parent.__init(self)
assert(p, 'p-norm not provided')
assert(p > 0, p..'-norm not supported')
self.p =p
self.eps = eps or 1e-10

end

function ChannelNormalization:updateOutput(input) -- done

assert(input:dim()>= 3, 'only data including feature dim supported')

local input_size = input:size()
local featureDim
if input:dim() == 4 then featureDim  = 2
elseif input:dim() == 3 then featureDim = 1
else assert(nil,'you need to write down code over 4 dim')
end

self._output = self._output or input.new()
self.norm = self.norm or input.new()
--self.normp = self.normp or input.new()
self._repeat = self._repeat or input.new()
self._expand = self._expand or input.new()
self._output:resizeAs(input)


self.norm = torch.norm(input,self.p,featureDim)
--self.normp:copy( self.norm):pow(self.p)
self.norm:add(self.eps)

self._expand:expandAs(self.norm,self._output)

if torch.type(input) == 'torch.CudaTensor' then
self._repeat:resizeAs(self._expand):copy(self._expand)
self._output:cdiv(input,self._repeat)
        else
self._output:cdiv(input,self._expand)
end
self.output:view(self._output,input_size)

return self.output
end


function ChannelNormalization:updateGradInput(input, gradOutput)

assert(input:dim()>=3 , 'only data including feature dim supported')
assert(gradOutput:dim() >=3 , 'only data including feature dimi supported')

local input_size = input:size()

local n , d
if input:dim() == 4 then
n = input:size(1) -- batch size
d = input:size(2) -- feature dim
elseif input:dim() == 3 then
n = 1
d = input:size(1)
end

self._gradInput = self._gradInput  or input.new()
--self._dxNorm = self._dxNorm or input.new()
self.buffer = self.buffer or input.new() -- differential upper
self.buffer2 = self.buffer2 or input.new() -- differential lower

if self.p ~= 2 then
  
  assert(nil,'not yet provided')
elseif self.p ==2 then
self.buffer:add(self.norm:expandAs(input),1)
    self.buffer:cinv()
    self.buffer2:pow(self.norm:expandAs(input),-3/2)
self.buffer2:cmul(input/2)
    else
--[[
    self.buffer = (self.norm:expandAs(input))
  self.buffer:cinv()
  self.buffer2 = input:clone()
  self.buffer2:pow(self.p)
  self.buffer2:cmul(self.normp:expandAs(input):pow((1-self.p)/self.p))
  self.buffer2:cdiv(self.buffer)
  self.buffer2:cdiv(self.buffer)
]]--
end

self._gradInput = self.buffer -  self.buffer2
--self._gradInput = self._gradInput * self.stdnorm
self._gradInput:cmul(gradOutput)
self.gradInput:view(self._gradInput, input_size)

return self.gradInput

end

function ChannelNormalization:__tostring__()
local s
-- different prints if the norm is integer
if self.p % 1 == 0 then
s = '%s(%d)'
else 
s = '%s(%f)' 
end
return string.format(s,torch.type(self),self.p)
end


function ChannelNormalization:type(type, tensorCache)
self._indices = nil
parent.type(self, type, tensorCache)
return self
end

function ChannelNormalization:clearState()
nn.utils.clear(self, 
{'_output',
--'_indices',
'_gradInput',
'buffer',
'norm',
--'normp',
'buffer2', '_expand', '_repeat'
})
return parent.clearState(self)
end
