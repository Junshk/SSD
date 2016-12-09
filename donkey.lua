require 'ffi'
Threads = require 'threads'
Threads.serialization('threads.sharedserialize')
local nDonkey = 2
local manualseed  = os.time()
do
  if nDonkey >0 then
donkeys = Threads(nDonkey,
                  function()
                  require 'cutorch'
                  require 'cunn'
                  require 'cudnn'
                  require 'FetchData'
                  --pretrain = torch.load('pretrain.net')
                  end,
                  function(idx)
                  local seed = manualseed + idx
                  torch.manualSeed(seed)
                  print('st donkey id, seed',idx,seed)
                  end
                 
  );

else donkeys = {}
      function donkeys:addjob(f1, f2) f2(f1()) end
            function donkeys:synchronize() end
            end



end
