
-- we test on PASCAL VOC 2012
-- training dataset is comprise of VOC2007 trainval, test, VOC2012 trainval

local savefolder = 'VOCdevkit/'
if paths.dirp(savefolder)~= true then os.execute('mkdir '..savefolder) end

        -- voc 2007 trainval
  if paths.filep('VOCtrainvl_06-Nov-2007.tar') ~=true then
      os.execute('wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar') end

                -- voc 2007 test
      if paths.filep('VOCtest_06-Nov-2007.tar')~=true then
         os.execute('wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar')end
      if paths.filep('VOCtestnoimgs_06-Nov-2007.tar') ~=true then 
         os.execute('wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtestnoimgs_06-Nov-2007.tar') -- annotation
             end
            -- voc 2012 trainval
    if paths.filep('VOCtrainval_11-May-2012.tar') ~= true then
       os.execute('wget http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar')
              end
                                        -- voc 2012 test
                                        --os.execute('')  -- ?? can i get this files?
                                        --os.execute('')

                      --unzip
                  os.execute('tar -xvf VOCtest_06-Nov-2007.tar '..savefolder)
                  os.execute('tar -xvf VOCtrainval_06-Nov-2007.tar '..savefolder)
                  os.execute('tar -xvf VOCtrainval_11-May-2012.tar '..savefolder)

