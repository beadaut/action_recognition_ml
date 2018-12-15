%{
MIT License

Copyright (c) 2017 Josué Rocha Lima - Centro Federal de Educação
Tecnologica de Minas Gerais - josuerocha@me.com - github.com/josuerocha

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
%}
function [depthMap] = loadDepthMap(path)
%This function reads a depth image from MSR Action3D dataset  
%Input:  
%   path - location of the bin file  
%Output:    
%   depthImg - depth image  

%% READING FROM FILE  
fid = fopen(path); %Assign file reading object 
[dims, numFrames] = readHeader(fid);
fileData = fread(fid, 'uint32');
fclose(fid); %close the stream  

%% CONVERTING TO DEPTH MAP FORMAT  

depth = double(fileData); %convert to double for imaging  
depthCountPerMap = prod(dims);

depthMap = cell(1,numFrames);
for i=1 : numFrames
    
    currentDepthData = depth(1:depthCountPerMap);
    depth(1:depthCountPerMap) = [];
%     depthMap{i} = reshape(currentDepthData, dims(1), dims(2))'; %reshape depth into matrix
    depthMap{i} = currentDepthData;
    
    
end

end  

function [dims,numFrames] = readHeader(fid)
      
     numFrames = typecast(uint8(fread(fid,4)), 'uint32');
     dims(1) = typecast(uint8(fread(fid,4)), 'uint32');
     dims(2) = typecast(uint8(fread(fid,4)), 'uint32');
end