require 'image'
torch.setdefaulttensortype('torch.FloatTensor')

local xml = require 'xml'


function pascal_save(opt)
local data = {}
data.folder, data.imgname, data.image, data.object = {},{},{},{}

local root = opt.root or 'VOCdevkit/'

local idxMax = opt.idxMax or 1000
local year = opt.year or '2007'
local folder = self.root .. '/VOC'..year

local saveName = 'VOC'..year ..'_'
if idxMax ~= math.max then saveName = saveName .. idxMax end

print(folder)

local idx = 1
for image_name in paths.iterfiles(folder.. '/JPEGImages/') do

	if idx <= idxMax then

		if idx%1000==0 then print(idx) end
		local imgname = image_name:sub(1,-5)
		local annot = xml.loadpath(folder .. '/Annotations/' .. imgname .. '.xml')
		
		data.folder[idx] = folder
		data.imgname[idx] = imgname
		data.image[idx] = image.load(folder .. '/JPEGImages/' .. imgname .. '.jpg')
		data.image[idx] = (data.image[idx]*255):byte()
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
							object.bbox = torch.Tensor({v[2][1],v[4][1],v[1][1],v[3][1]}) -- xmin, ymin,xmax,ymax
					          else assert(0,'wrong xml info')
            end
					end
				end
			end
		end


	end

	idx = idx + 1
	collectgarbage()
end


torch.save(saveName,data)

end
function ImgInfo()
local Info = { }
local year = {'2007','2012'}

for k,year_path in pairs(year) do
  local path = 'VOCdevkit/VOC'..year_path..'/'
for image_name in paths.iterfiles(path..'/JPEGImages/') do
table.insert(Info,{path = path, image_name = image_name})

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
		data.image[idx] = (data.image[idx]*255):byte()
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
							object.bbox = torch.Tensor({v[2][1],v[4][1],v[1][1],v[3][1]}) -- xmin, ymin,xmax,ymax
					          else assert(0,'wrong xml info')
            end
					end
				end
			end
	


	end

--	idx = idx + 1
	collectgarbage()



return data

end


