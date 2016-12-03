require 'option'
require 'image'
require 'make_net'
local pretrain =torch.load('pretrain.net')
local net = torch.load('model/'..Option.netname..'_intm.net')

for i= 1,100 do

print('num',i)

data = torch.load('data/SSDdata_'..i..'.t7')
image.save(i..'check.jpg',data.input:squeeze())
local t = net:forward(pretrain:forward(data.input:cuda())):float()

local _,cls =torch.max(t[{{},{1,21}}],2)


local cmp = torch.cat(cls:float(),data.target[{{},{5}}],1)


--print(cmp:size())
for iter = 1,cmp:size(3) do

if cmp[{1,1,{iter}}]:squeeze()~=21 
  and cmp[{1,1,{iter}}]:squeeze()==cmp[{2,1,{iter}}]:squeeze()
  then
  local class =cmp[{{},1,iter}]
  print(iter, class,t[{1,class,iter}])
end
  end

--print(data.target:size(),t:size())
--print(data.target[idx:expand(data.target:size())],t[idx:expand(t:size())])

  end
