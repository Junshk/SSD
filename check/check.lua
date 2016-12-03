require 'option'

require 'test'

require 'make_data'



i, t = patchFetch(1,img_Info_table)

for iter = 1, i:size(1) do
--print(iter,torch.sum(torch.ne( t[{{iter},{5}}],21)))
--print(t[{{iter},{5}}])
--assert(nil)
--print(t[{{iter},{}}][torch.ne(t[{{iter},{5}}]:expand(t[{{iter},{}}]:size()),21)]:view(5,-1))

  end


test_tensor(t,i,'validation/')
