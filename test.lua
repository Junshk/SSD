require 'cunn'
require 'cudnn'
require 'etc'
require 'image'

local list = ImgInfo2()
local net = torch.load('model/vggSSDnet_intm.t7')

for iter = 1, #list do

local imagename

local img = image.load(imagename)
img = image:scale(img,500,500)

local result_vector = net:forward(img:cuda())







local result_box = nms()

compare

end


