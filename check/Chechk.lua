require 'option'
require 'test'

require 'make_net'
i= image.load('VOCdevkit/VOC2012/JPEGImages/2008_001722.jpg')
i = image.scale(i,500,500)
i=i:view(1,3,500,500)
a=torch.load('model/'..Option.netname..'_intm.net')
p=torch.load('pretrain.net')
n=nn.Sequential()

n:add(p)
n:add(a)
print(n)
oo=n:forward(i:cuda())


test_tensor(oo:float(),i:float(),'validation/')


