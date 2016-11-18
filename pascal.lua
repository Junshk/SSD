require 'image'
torch.setdefaulttensortype('torch.FloatTensor')

local xml = require 'xml'

function trainInfo()
local list = {}
local t2012 = 'VOCdevkit/VOC2012/ImageSets/Main/trainval.txt'
local t2007 = 'VOCdevkit/VOC2007/ImageSets/Main/trainval.txt'


local f = assert(io.open(t2012,'r'))

io.input(f)
while true do
  local img ={}
  local line = io.read()
  if line ==nil then break end
   
    name =  line..'.jpg'
    table.insert(list,{path='VOCdevkit/VOC2012/',image_name = name})

end

 f = assert(io.open(t2007,'r'))

io.input(f)
while true do
  local img ={}
  local line = io.read()
  if line ==nil then break end
    
    name =  line..'.jpg'
    table.insert(list,{path='VOCdevkit/VOC2007/',image_name = name})

end









return list
end
function ImgInfo(folder)

local Info = {}
local folder = folder 

if folder == nil then 
local path = 'VOCdevkit/VOC'
folder = {path..'2007/',path..'2012/'}
end

for k, folder_path in pairs(folder) do
  for image_name in paths.iterfiles(folder_path..'/JPEGImages/') do
    table.insert(Info,{path=folder_path,image_name=image_name})
  end
end


return Info
end


function pascal_loadAImage(opt)
local data = {}
data.folder, data.imgname, data.image, data.object = {},{},{},{}
local idx = 1

local path = opt.info.path


--for image_name in paths.iterfiles(folder.. '/JPEGImages/') do
local image_name = opt.info.image_name

--	if idx <= idxMax then

--		if idx%1000==0 then print(idx) end
		local imgname = image_name:sub(1,-5)
--print(imgname)
local annot = xml.loadpath(path .. '/Annotations/' .. imgname .. '.xml')
		
		data.folder[idx] = folder
		data.imgname[idx] = imgname
		data.image[idx] = image.load(path .. '/JPEGImages/' .. imgname .. '.jpg')
		data.image[idx] = (data.image[idx]*255)
    data.object[idx] = {}
		
		local iObject = 0
		for _,item in pairs(annot) do
			if item.xml == 'object' then
				iObject  = iObject + 1
				data.object[idx][iObject] = {}
				local object = data.object[idx][iObject]

				for k,v in pairs(item) do
					if type(v)=='table' then
						if  v.xml=='name' then
							object.class = v[1]
						elseif v.xml=='pose' then
							object.pose = v[1]
						elseif v.xml=='truncated' then
							object.truncated = v[1]
						elseif v.xml=='difficult' then
							object.difficult = v[1]
						elseif v.xml=='bndbox' then
							local BoundBox = torch.Tensor(4):fill(0)
              for k_box,v_box in pairs(v) do
                if v_box.xml == 'xmin' then BoundBox[{1}] =v_box[1]     
                elseif v_box.xml == 'ymin' then BoundBox[{2}] =v_box[1]
                elseif v_box.xml == 'xmax' then BoundBox[{3}] =v_box[1]
                elseif v_box.xml == 'ymax' then BoundBox[{{4}}] =v_box[1]
                end
              end
               object.bbox = BoundBox -- xmin, ymin,xmax,ymax
					  
             --else assert(nil,'wrong xml info')
            end
					end
				end
			end
	


	end

--	idx = idx + 1
	collectgarbage()



return data

end


