require 'image'
local xml = require 'xml';

local num2007 = 0
local path_2007 = 'VOC2007/'
local path_2012 = 'VOC2012/'
torch.setdefaulttensortype('torch.FloatTensor')

local data = {}
data.folder, data.name, data.image, data.object = {},{},{},{}

print('VOC 2007....')
local path = 'VOCdevkit/' .. path_2007
local idx = 1
for image_name in paths.iterfiles(path .. 'JPEGImages/') do

	if idx <= 1000 then

		if idx%1000==0 then print(idx) end
		local name = image_name:sub(1,-5)
		local annot = xml.loadpath(path .. 'Annotations/' .. name .. '.xml')
		
		data.folder[idx] = path_2007
		data.name[idx] = name
		data.image[idx] = image.load(path .. 'JPEGImages/' .. name .. '.jpg')
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
							object.bbox = torch.Tensor({v[1][1],v[2][1],v[4][1],v[4][1]})
						end
					end
				end
			end
		end


	end

	idx = idx + 1
	collectgarbage()
end
print('#2007 images : ' .. idx-1)
--[[
print('VOC 2012....')
path = 'VOCdevkit/' .. path_2012
for image_name in paths.iterfiles(path .. 'JPEGImages/') do
	if idx%1000==0 then print(idx) end
	local name = image_name:sub(1,-5)

	if name:sub(1,4)~='2007' then
		local annot = xml.loadpath(path .. 'Annotations/' .. name .. '.xml')
		
		data.folder[idx] = path_2007
		data.name[idx] = name
		data.image[idx] = image.load(path .. 'JPEGImages/' .. name .. '.jpg')
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
							object.bbox = torch.Tensor({v[1][1],v[2][1],v[4][1],v[4][1]})
						end
					end
				end
			end
		end

		idx = idx + 1
	else
		num2007 = num2007 + 1
	end
	collectgarbage()
end
--]]
print('End. Total # iamges = ' .. idx-1)
print('There were ' .. num2007 .. ' 2007 images in 2012 dataset')

print('saving...')
--torch.save('data.t7',data)
torch.save('data_sub.t7',data)
