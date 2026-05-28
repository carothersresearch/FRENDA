# from https://github.com/divideconcept/fastnumpyio/blob/main/fastnumpyio.py
import sys
import numpy as np
import numpy.lib.format
import struct

def save(file, array):
    magic_string=b"\x93NUMPY\x01\x00v\x00"
    header=bytes(("{'descr': '"+array.dtype.descr[0][1]+"', 'fortran_order': False, 'shape': "+str(array.shape)+", }").ljust(127-len(magic_string))+"\n",'utf-8')
    if type(file) == str:
        with open(file,"wb") as f:
            f.write(magic_string)
            f.write(header)
            f.write(array.data)
    del array, header, magic_string

def pack(array):
    size=len(array.shape)
    return bytes(array.dtype.byteorder.replace('=','<' if sys.byteorder == 'little' else '>')+array.dtype.kind,'utf-8')+array.dtype.itemsize.to_bytes(1,byteorder='little')+struct.pack(f'<B{size}I',size,*array.shape)+array.data

def load(file):
    if type(file) == str:
        with open(file,"rb") as f:
            header = f.read(128)
            if not header:
                return None
            descr = str(header[19:25], 'utf-8').replace("'","").replace(" ","")
            shape = tuple(int(num) for num in str(header[60:120], 'utf-8').replace(', }', '').replace('(', '').replace(')', '').split(','))
            datasize = numpy.lib.format.descr_to_dtype(descr).itemsize
            for dimension in shape:
                datasize *= dimension
            array = np.ndarray(shape, dtype=descr, buffer=f.read(datasize))
            
    return array

def unpack(data):
    dtype = str(data[:2],'utf-8')
    dtype += str(data[2])
    size = data[3]
    shape = struct.unpack_from(f'<{size}I', data, 4)
    datasize=data[2]
    for dimension in shape:
        datasize *= dimension
    return np.ndarray(shape, dtype=dtype, buffer=data[4+size*4:4+size*4+datasize])