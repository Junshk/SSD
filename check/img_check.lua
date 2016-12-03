require 'image'

local start = 1
local endi= 1000

for iter = start, endi do

local data = torch.load('data/SSDdata_'..iter..'.t7')
local img = data.input:squeeze()
img[{{1}}]:add(104)
img[{{2}}]:add(117)
img[{{3}}]:add(123)
img = img/255
image.save('checkimg/img_'..iter..'.jpg',data.input:squeeze())

  end
