from django.shortcuts import render
from django.http import HttpResponse
import types
from django.core.files.storage import FileSystemStorage
from .models import encrypted_data
from django.contrib.auth.models import User, auth
from django.db.models import Q



#for aes implementation
import unittest
import base64
import os
#pip install py3rijndael
from py3rijndael import Rijndael

import random
import string


# Create your views here.

def index(request):
    return render(request,'index.html')

def Encrypt_and_send(request):
    letters=string.ascii_letters
    random_saveas = ''.join(random.choice(letters) for i in range(8))
    print(random_saveas)
    #key_changed =  base64.b64encode(key)
    users_details=User.objects.all()
    return render(request,'Encrypt_and_send.html',{'users_details':users_details,'random_saveas':random_saveas})#,'key':key,'key_changed':key_changed

def Encrypt_and_download(request):
	users_details=User.objects.all()
	return render(request,'Encrypt_and_download.html')
    
def decbtn(request):
    return render(request,'Decrypt.html')


# using Q objects, F objects , 
def sent(request):
	user_firstname='none'
	if request.user.is_authenticated:
		#print(User.objects.get(username='pbchandra3'))
		user_username=request.user.username
		#print(request.user.username)
		test=encrypted_data.objects.filter(sendby=user_username).order_by('-id')
		#test2=encrypted_data.objects.filter(Q(Q(secret_key=8) | Q(secret_key=22) & Q(msg='jp}~ivi(m(vmml(|w(jm(ni{|'))).query
        
		return render(request,'items.html',{'test':test})
	else:
		#print("else")
		return render(request,'index.html')

def received(request):
	user_firstname='none'
	if request.user.is_authenticated:
		#print(User.objects.get(username='pbchandra3'))
		user_username=request.user.username
		#print(request.user.username)
		test=encrypted_data.objects.filter(sendto=user_username).order_by('-id')
		#test=encrypted_data.objects.filter(Q(Q(secret_key=8) | Q(secret_key=22) & Q(msg='jp}~ivi(m(vmml(|w(jm(ni{|'))).query
		return render(request,'items.html',{'test':test})
	else:
		#print("else")
		return render(request,'index.html')

def user_decrypt(request):
	test_id=request.POST['p_id']
	test=encrypted_data.objects.filter(Q(id=test_id))
	return render(request,'user_decrypt.html',{'test': test})


#stegnography actual code
# Python program implementing Image Steganography

# PIL module is used to extract
# pixels of image and modify it
from PIL import Image

# Convert encoding data into 8-bit binary
# form using ASCII value of characters
def genData(data):

		# list of binary codes
		# of given data
		newd = []

		for i in data:
			newd.append(format(ord(i), '08b'))
		return newd

# Pixels are modified according to the
# 8-bit binary data and finally returned
def modPix(pix, data):

	datalist = genData(data)
	lendata = len(datalist)
	imdata = iter(pix)

	for i in range(lendata):

		# Extracting 3 pixels at a time
		pix = [value for value in imdata.__next__()[:3] +
								imdata.__next__()[:3] +
								imdata.__next__()[:3]]

		# Pixel value should be made
		# odd for 1 and even for 0
		for j in range(0, 8):
			if (datalist[i][j] == '0' and pix[j]% 2 != 0):
				pix[j] -= 1

			elif (datalist[i][j] == '1' and pix[j] % 2 == 0):
				pix[j] += 1
					
				# pix[j] -= 1

		# Eighth pixel of every set tells
		# whether to stop ot read further.
		# 0 means keep reading; 1 means thec
		# message is over.
		if (i == lendata - 1):
			if (pix[-1] % 2 == 0):
				if(pix[-1] != 0):
					pix[-1] -= 1
				else:
					pix[-1] += 1

		else:
			if (pix[-1] % 2 != 0):
				pix[-1] -= 1

		pix = tuple(pix)
		yield pix[0:3]
		yield pix[3:6]
		yield pix[6:9]

def encode_enc(newimg, data):
	w = newimg.size[0]
	(x, y) = (0, 0)

	for pixel in modPix(newimg.getdata(), data):

		# Putting modified pixels in the new image
		newimg.putpixel((x, y), pixel)
		if (x == w - 1):
			x = 0
			y += 1
		else:
			x += 1
# Encode data into image
def encryptanddownload(request):
	uploaded_file = request.FILES["nrmlimg"]
	fs=FileSystemStorage()
	fs.save(uploaded_file.name,uploaded_file)
	path = r'C:\Users\hp\Projects\my_first_project\media\\'+ uploaded_file.name;
	img = uploaded_file.name
	image = Image.open(path, 'r')
	data = request.POST["msg"]
	if (len(data) == 0):
		raise ValueError('Data is empty')

	cryptographic_algorithm=request.POST["cryptographic_algorithm"]
	
	if cryptographic_algorithm =='1':
		secret_key=request.POST["secret_key"]        
		data=ceasar_encrypt(data,secret_key)
	
	if cryptographic_algorithm =='2':
		secret_key=request.POST["secret_key_des"]        
		count=0
		while len(data) % 8 != 0:
			data = data + " "
			count= count+1
		
		d = des()
		data=d.encrypt(secret_key,data)
	
	newimg = image.copy()
	encode_enc(newimg, data)

	name = request.POST["saveasname"]
	extenstion = request.POST["extension"]
	new_img_name = name+extenstion
	path = r'C:\Users\hp\Projects\my_first_project\static\\'+new_img_name;
	
	newimg.save(path, str(new_img_name.split(".")[1].upper()))
	
	return render(request,'results.html',{'test':new_img_name,'filename':new_img_name})
# Encode data into image
def encryptandsend(request):
	uploaded_file = request.FILES["nrmlimg"]
	fs=FileSystemStorage()
	fs.save(uploaded_file.name,uploaded_file)
	path = r'C:\Users\hp\Projects\my_first_project\media\\'+ uploaded_file.name;
	img = uploaded_file.name
	image = Image.open(path, 'r')

	data = request.POST["msg"]
	sendby = request.POST["sendby"]
	sendto = request.POST["sendto"]
	if (len(data) == 0):
		raise ValueError('Data is empty')
	
	cryptographic_algorithm=request.POST["cryptographic_algorithm"]

	if cryptographic_algorithm =='NoEncryption':
		secret_key=''	
	
	if cryptographic_algorithm =='ceasar_encrypt':
		secret_key=request.POST["secret_key_ceasar"]
		data=ceasar_encrypt(data,secret_key)
	if cryptographic_algorithm =='aes':
		plain_text = data.encode('utf-8')
		padded_text = plain_text.ljust(32, b'\x1b')
       
		secret_key=os.urandom(32)      
		rijndael_key = Rijndael(secret_key, block_size=32)
		cipher = rijndael_key.encrypt(padded_text)
		cipher_text = base64.b64encode(cipher)                
		data=cipher   
		cipher_text=cipher_text.decode('utf-8')
		data=cipher_text        
        #padded_text = plain_text.ljust(32, b'\x1b')        
			
	if cryptographic_algorithm =='des':
		secret_key=request.POST["secret_key_des"]        
		count=0
		while len(data) % 8 != 0:
			data = data + " "
			count= count+1
		d = des()
		data=d.encrypt(secret_key,data);

	print(data)
	newimg = image.copy()
	encode_enc(newimg, data)

	name = request.POST["saveasname"]
	extenstion = request.POST["extension"]
	new_img_name = name+extenstion
	path = r'C:\Users\hp\Projects\my_first_project\static\\'+new_img_name;
	
	newimg.save(path, str(new_img_name.split(".")[1].upper()))
	posting = encrypted_data(sendby=sendby,sendto=sendto,nrmlimg=uploaded_file.name,msg=data,sensitivity=cryptographic_algorithm,secret_key=secret_key,encryimg=new_img_name)
	posting.save()
	return render(request,'results.html',{'test':new_img_name,'filename':new_img_name})

#direct decryption 
def decryption(request):
    uploaded_file = request.FILES["stignoimg"]
    fs=FileSystemStorage()
    fs.save(uploaded_file.name,uploaded_file)
    img = r'C:\Users\hp\Projects\my_first_project\media\\'+ uploaded_file.name
    image = Image.open(img, 'r')
    data = ''
    imgdata = iter(image.getdata())
 
    while (True):
        pixels = [value for value in imgdata.__next__()[:3] +
                                imgdata.__next__()[:3] +
                                imgdata.__next__()[:3]]
 
        # string of binary data
        binstr = ''
 
        for i in pixels[:8]:
            if (i % 2 == 0):
                binstr += '0'
            else:
                binstr += '1'
 
        data += chr(int(binstr, 2))
        cryptographic_algorithm=request.POST["cryptographic_algorithm"]
        secret_key=request.POST["secret_key"]
        plain_text=''
        if (pixels[-1] % 2 != 0):
            if cryptographic_algorithm == '0':
                plain_text = data            
            if cryptographic_algorithm == '1':
                print(data)
                plain_text=ceasar_decrypt(data,secret_key)
            if cryptographic_algorithm == '2':
                secret_key=request.POST["secret_key_des"]
                d = des()
                plain_text = d.decrypt(secret_key,data)
                             
            return render(request,'results2.html',{'result': plain_text})


def userdecryption(request):
    uploaded_file = request.POST["stignoimg"]

    img = r'C:\Users\hp\Projects\my_first_project\static' + "\\"+uploaded_file
	
    image = Image.open(img, 'r')
 
    data = ''
    imgdata = iter(image.getdata())
 
    while (True):
        pixels = [value for value in imgdata.__next__()[:3] +
                                imgdata.__next__()[:3] +
                                imgdata.__next__()[:3]]
 
        # string of binary data
        binstr = ''
 
        for i in pixels[:8]:
            if (i % 2 == 0):
                binstr += '0'
            else:
                binstr += '1'
 
        data += chr(int(binstr, 2))
        cryptographic_algorithm=request.POST["cryptographic_algorithm"]
        secret_key=request.POST["secret_key"]
        if (pixels[-1] % 2 != 0):
            if cryptographic_algorithm == 'NoEncryption':
                plain_text = data            
            if cryptographic_algorithm =='ceasar_encrypt':
                plain_text=ceasar_decrypt(data,secret_key)

            if cryptographic_algorithm == 'des':
                secret_key=request.POST["secret_key_des"]
                d = des()
                plain_text = d.decrypt(secret_key,data)
            if cryptographic_algorithm == 'aes':
                test_id=request.POST['p_id'] 
                #test=encrypted_data.objects.values_list('secret_key').filter(Q(id=test_id))
                test=encrypted_data.objects.filter(Q(id=test_id))   
                for t in test:
                    print(t.secret_key)
                    key=t.secret_key
                #test=encrypted_data.objects.filter(Q(id=test_id))                 
                #test=encrypted_data.objects.filter(Q(id=test_id))                                                                               
                print(test)
                key=eval(key)
                rijndael_key = Rijndael(key, block_size=32)
                cipher= data.encode('utf-8')
                cipher = base64.b64decode(cipher)
                #cipher = b'\xacNg\x97\xea\xd5&\xf3\xae\xad;:q\xe7\x86\x94\x98y\x1b\x8f=<Hs\x1b\xcd\xdf~h\xee.T'           
                #rijndael_key = Rijndael(key, block_size=32)                                                
                plain_text=rijndael_key.decrypt(cipher)
                #plain_text = data
                plain_text = plain_text.decode('utf-8')
                plain_text_new=''

                for z in plain_text:
                    if z !='':
                        plain_text_new = plain_text_new + z                
            return render(request,'results2.html',{'result': plain_text_new})



#cryptography
#ceaser cipher

def ceasar_encrypt(plain_text,key):
    cipher_text=''
    for char in plain_text:
        char = ord(char)+ int(key);#gets the ascii value of string and add's key to it
        cipher_text=cipher_text+chr(char)
    return cipher_text

def ceasar_decrypt(cipher_text,key):
    plain_text=''
    print(cipher_text)
    for char in cipher_text:
        char = ord(char) - int(key) #gets the ascii value of string and substracts key from it
        plain_text = plain_text+chr(char)
        print(plain_text)
    return plain_text

    return plain_text




#DES

#-*- coding: utf8 -*-

#Initial permut matrix for the datas
PI = [58, 50, 42, 34, 26, 18, 10, 2,
      60, 52, 44, 36, 28, 20, 12, 4,
      62, 54, 46, 38, 30, 22, 14, 6,
      64, 56, 48, 40, 32, 24, 16, 8,
      57, 49, 41, 33, 25, 17, 9, 1,
      59, 51, 43, 35, 27, 19, 11, 3,
      61, 53, 45, 37, 29, 21, 13, 5,
      63, 55, 47, 39, 31, 23, 15, 7]

#Initial permut made on the key
CP_1 = [57, 49, 41, 33, 25, 17, 9,
        1, 58, 50, 42, 34, 26, 18,
        10, 2, 59, 51, 43, 35, 27,
        19, 11, 3, 60, 52, 44, 36,
        63, 55, 47, 39, 31, 23, 15,
        7, 62, 54, 46, 38, 30, 22,
        14, 6, 61, 53, 45, 37, 29,
        21, 13, 5, 28, 20, 12, 4]

#Permut applied on shifted key to get Ki+1
CP_2 = [14, 17, 11, 24, 1, 5, 3, 28,
        15, 6, 21, 10, 23, 19, 12, 4,
        26, 8, 16, 7, 27, 20, 13, 2,
        41, 52, 31, 37, 47, 55, 30, 40,
        51, 45, 33, 48, 44, 49, 39, 56,
        34, 53, 46, 42, 50, 36, 29, 32]

#Expand matrix to get a 48bits matrix of datas to apply the xor with Ki
E = [32, 1, 2, 3, 4, 5,
     4, 5, 6, 7, 8, 9,
     8, 9, 10, 11, 12, 13,
     12, 13, 14, 15, 16, 17,
     16, 17, 18, 19, 20, 21,
     20, 21, 22, 23, 24, 25,
     24, 25, 26, 27, 28, 29,
     28, 29, 30, 31, 32, 1]

#SBOX
S_BOX = [
         
[[14, 4, 13, 1, 2, 15, 11, 8, 3, 10, 6, 12, 5, 9, 0, 7],
 [0, 15, 7, 4, 14, 2, 13, 1, 10, 6, 12, 11, 9, 5, 3, 8],
 [4, 1, 14, 8, 13, 6, 2, 11, 15, 12, 9, 7, 3, 10, 5, 0],
 [15, 12, 8, 2, 4, 9, 1, 7, 5, 11, 3, 14, 10, 0, 6, 13],
],

[[15, 1, 8, 14, 6, 11, 3, 4, 9, 7, 2, 13, 12, 0, 5, 10],
 [3, 13, 4, 7, 15, 2, 8, 14, 12, 0, 1, 10, 6, 9, 11, 5],
 [0, 14, 7, 11, 10, 4, 13, 1, 5, 8, 12, 6, 9, 3, 2, 15],
 [13, 8, 10, 1, 3, 15, 4, 2, 11, 6, 7, 12, 0, 5, 14, 9],
],

[[10, 0, 9, 14, 6, 3, 15, 5, 1, 13, 12, 7, 11, 4, 2, 8],
 [13, 7, 0, 9, 3, 4, 6, 10, 2, 8, 5, 14, 12, 11, 15, 1],
 [13, 6, 4, 9, 8, 15, 3, 0, 11, 1, 2, 12, 5, 10, 14, 7],
 [1, 10, 13, 0, 6, 9, 8, 7, 4, 15, 14, 3, 11, 5, 2, 12],
],

[[7, 13, 14, 3, 0, 6, 9, 10, 1, 2, 8, 5, 11, 12, 4, 15],
 [13, 8, 11, 5, 6, 15, 0, 3, 4, 7, 2, 12, 1, 10, 14, 9],
 [10, 6, 9, 0, 12, 11, 7, 13, 15, 1, 3, 14, 5, 2, 8, 4],
 [3, 15, 0, 6, 10, 1, 13, 8, 9, 4, 5, 11, 12, 7, 2, 14],
],  

[[2, 12, 4, 1, 7, 10, 11, 6, 8, 5, 3, 15, 13, 0, 14, 9],
 [14, 11, 2, 12, 4, 7, 13, 1, 5, 0, 15, 10, 3, 9, 8, 6],
 [4, 2, 1, 11, 10, 13, 7, 8, 15, 9, 12, 5, 6, 3, 0, 14],
 [11, 8, 12, 7, 1, 14, 2, 13, 6, 15, 0, 9, 10, 4, 5, 3],
], 

[[12, 1, 10, 15, 9, 2, 6, 8, 0, 13, 3, 4, 14, 7, 5, 11],
 [10, 15, 4, 2, 7, 12, 9, 5, 6, 1, 13, 14, 0, 11, 3, 8],
 [9, 14, 15, 5, 2, 8, 12, 3, 7, 0, 4, 10, 1, 13, 11, 6],
 [4, 3, 2, 12, 9, 5, 15, 10, 11, 14, 1, 7, 6, 0, 8, 13],
], 

[[4, 11, 2, 14, 15, 0, 8, 13, 3, 12, 9, 7, 5, 10, 6, 1],
 [13, 0, 11, 7, 4, 9, 1, 10, 14, 3, 5, 12, 2, 15, 8, 6],
 [1, 4, 11, 13, 12, 3, 7, 14, 10, 15, 6, 8, 0, 5, 9, 2],
 [6, 11, 13, 8, 1, 4, 10, 7, 9, 5, 0, 15, 14, 2, 3, 12],
],
   
[[13, 2, 8, 4, 6, 15, 11, 1, 10, 9, 3, 14, 5, 0, 12, 7],
 [1, 15, 13, 8, 10, 3, 7, 4, 12, 5, 6, 11, 0, 14, 9, 2],
 [7, 11, 4, 1, 9, 12, 14, 2, 0, 6, 10, 13, 15, 3, 5, 8],
 [2, 1, 14, 7, 4, 10, 8, 13, 15, 12, 9, 0, 3, 5, 6, 11],
]
]

#Permut made after each SBox substitution for each round
P = [16, 7, 20, 21, 29, 12, 28, 17,
     1, 15, 23, 26, 5, 18, 31, 10,
     2, 8, 24, 14, 32, 27, 3, 9,
     19, 13, 30, 6, 22, 11, 4, 25]

#Final permut for datas after the 16 rounds
PI_1 = [40, 8, 48, 16, 56, 24, 64, 32,
        39, 7, 47, 15, 55, 23, 63, 31,
        38, 6, 46, 14, 54, 22, 62, 30,
        37, 5, 45, 13, 53, 21, 61, 29,
        36, 4, 44, 12, 52, 20, 60, 28,
        35, 3, 43, 11, 51, 19, 59, 27,
        34, 2, 42, 10, 50, 18, 58, 26,
        33, 1, 41, 9, 49, 17, 57, 25]

#Matrix that determine the shift for each round of keys
SHIFT = [1,1,2,2,2,2,2,2,1,2,2,2,2,2,2,1]

def string_to_bit_array(text):#Convert a string into a list of bits
    array = list()
    for char in text:
        binval = binvalue(char, 8)#Get the char value on one byte
        array.extend([int(x) for x in list(binval)]) #Add the bits to the final list
    return array

def bit_array_to_string(array): #Recreate the string from the bit array
    res = ''.join([chr(int(y,2)) for y in [''.join([str(x) for x in _bytes]) for _bytes in  nsplit(array,8)]])   
    return res

def binvalue(val, bitsize): #Return the binary value as a string of the given size 
    binval = bin(val)[2:] if isinstance(val, int) else bin(ord(val))[2:]
    if len(binval) > bitsize:
        raise "binary value larger than the expected size"
    while len(binval) < bitsize:
        binval = "0"+binval #Add as many 0 as needed to get the wanted size
    return binval

def nsplit(s, n):#Split a list into sublists of size "n"
    return [s[k:k+n] for k in range(0, len(s), n)]

ENCRYPT=1
DECRYPT=0

class des():
    def __init__(self):
        self.password = None
        self.text = None
        self.keys = list()
        
    def run(self, key, text, action=ENCRYPT, padding=False):
        if len(key) < 8:
            raise "Key Should be 8 bytes long"
        elif len(key) > 8:
            key = key[:8] #If key size is above 8bytes, cut to be 8bytes long
        
        self.password = key
        self.text = text
        
        if padding and action==ENCRYPT:
            self.addPadding()
        elif len(self.text) % 8 != 0:#If not padding specified data size must be multiple of 8 bytes
            raise "Data size should be multiple of 8"
        
        self.generatekeys() #Generate all the keys
        text_blocks = nsplit(self.text, 8) #Split the text in blocks of 8 bytes so 64 bits
        result = list()
        for block in text_blocks:#Loop over all the blocks of data
            block = string_to_bit_array(block)#Convert the block in bit array
            block = self.permut(block,PI)#Apply the initial permutation
            g, d = nsplit(block, 32) #g(LEFT), d(RIGHT)
            tmp = None
            for i in range(16): #Do the 16 rounds
                d_e = self.expand(d, E) #Expand d to match Ki size (48bits)
                if action == ENCRYPT:
                    tmp = self.xor(self.keys[i], d_e)#If encrypt use Ki
                else:
                    tmp = self.xor(self.keys[15-i], d_e)#If decrypt start by the last key
                tmp = self.substitute(tmp) #Method that will apply the SBOXes
                tmp = self.permut(tmp, P)
                tmp = self.xor(g, tmp)
                g = d
                d = tmp
            result += self.permut(d+g, PI_1) #Do the last permut and append the result to result
        final_res = bit_array_to_string(result)
        if padding and action==DECRYPT:
            return self.removePadding(final_res) #Remove the padding if decrypt and padding is true
        else:
            return final_res #Return the final string of data ciphered/deciphered
    
    def substitute(self, d_e):#Substitute bytes using SBOX
        subblocks = nsplit(d_e, 6)#Split bit array into sublist of 6 bits
        result = list()
        for i in range(len(subblocks)): #For all the sublists
            block = subblocks[i]
            row = int(str(block[0])+str(block[5]),2)#Get the row with the first and last bit
            column = int(''.join([str(x) for x in block[1:][:-1]]),2) #Column is the 2,3,4,5th bits
            val = S_BOX[i][row][column] #Take the value in the SBOX appropriated for the round (i)
            bin = binvalue(val, 4)#Convert the value to binary
            result += [int(x) for x in bin]#And append it to the resulting list
        return result
        
    def permut(self, block, table):#Permut the given block using the given table (so generic method)
        return [block[x-1] for x in table]
    
    def expand(self, block, table):#Do the exact same thing than permut but for more clarity has been renamed
        return [block[x-1] for x in table]
    
    def xor(self, t1, t2):#Apply a xor and return the resulting list
        return [x^y for x,y in zip(t1,t2)]
    
    def generatekeys(self):#Algorithm that generates all the keys
        self.keys = []
        key = string_to_bit_array(self.password)
        key = self.permut(key, CP_1) #Apply the initial permut on the key
        g, d = nsplit(key, 28) #Split it in to (g->LEFT),(d->RIGHT)
        for i in range(16):#Apply the 16 rounds
            g, d = self.shift(g, d, SHIFT[i]) #Apply the shift associated with the round (not always 1)
            tmp = g + d #Merge them
            self.keys.append(self.permut(tmp, CP_2)) #Apply the permut to get the Ki

    def shift(self, g, d, n): #Shift a list of the given value
        return g[n:] + g[:n], d[n:] + d[:n]
    
    def addPadding(self):#Add padding to the datas using PKCS5 spec.
        pad_len = 8 - (len(self.text) % 8)
        self.text += pad_len * chr(pad_len)
    
    def removePadding(self, data):#Remove the padding of the plain text (it assume there is padding)
        pad_len = ord(data[-1])
        return data[:-pad_len]
    
    def encrypt(self, key, text, padding=False):
        return self.run(key, text, ENCRYPT, padding)
    
    def decrypt(self, key, text, padding=False):
        return self.run(key, text, DECRYPT, padding)
    
# key = "12345678"
# text= "bhuvanaa"
# d = des()
# r = d.encrypt(key,text)
# r2 = d.decrypt(key,r)
# print("Ciphered: %r" % r)
# print("Deciphered: ", r2)
