import struct

def readImageFeatures(path):
  f = open(path, 'rb')
  while True:
    itemId = f.read(8)
    if itemId == '': break
    feature = struct.unpack('f'*4096, f.read(4*4096))
    print(itemId, feature)
    break

path = 'Behance_Image_Features.b'
readImageFeatures(path)