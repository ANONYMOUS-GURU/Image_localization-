# input_.py

import numpy as np
import cv2
import os


def make_val_test(train_csv,val_frac,test_frac):
	train_csv=train_csv.sample(frac=1.0)
	test_size=int(train_csv.shape[0]*test_frac)
	val_size=int(train_csv.shape[0]*val_frac)
	test=train_csv.iloc[:val_size,:]
	val=train_csv.iloc[val_size:test_size+val_size,:]
	train=train_csv.iloc[test_size+val_size:,:]

	return train,val,test

def make_batch(data):
	PATH=os.path.join(os.getcwd(),'images')
	i=0
	for x in range(data.shape[0]):
		im=data.iloc[x][0]
		img_addr=os.path.join(PATH,im)
		img=cv2.imread(img_addr)
		img=cv2.medianBlur(img,5)
		img=cv2.medianBlur(img,5)
		img=cv2.medianBlur(img,5)
		mask=cv2.Canny(img,100,100)
		mask=cv2.dilate(mask,(5,5),iterations = 10)
		mask=cv2.erode(mask,(5,5),iterations = 15)
		mask=np.reshape(mask,[480,640,-1])
		final_img=np.reshape(np.concatenate((img,mask),axis=2),[-1,480,640,4])
		
		labels=np.array([data.iloc[x][1],data.iloc[x][2],data.iloc[x][3],data.iloc[x][4]])
		if i>0:
			images=np.vstack((images,final_img))
			target=np.vstack((target,labels))
		else:
			images=final_img
			target=labels
		i+=1
	return images,target


def batch(train,batch_size):
	PATH=os.path.join(os.getcwd(),'images')
	i=0
	while i<train.shape[0]:
		x=0
		while x<batch_size and i<train.shape[0]:
			im=train.iloc[i][0]
			img_addr=os.path.join(PATH,im)
			img=cv2.imread(img_addr)
			img=cv2.medianBlur(img,5)
			img=cv2.medianBlur(img,5)
			img=cv2.medianBlur(img,5)
			mask=cv2.Canny(img,100,100)
			mask=cv2.dilate(mask,(5,5),iterations = 20)
			mask=cv2.erode(mask,(5,5),iterations = 15)
			mask=np.reshape(mask,[480,640,-1])
			final_img=np.concatenate((img,mask),axis=2)
			final_img=np.reshape(final_img,[1,480,640,4])
			labels=np.array([train.iloc[i][1],train.iloc[i][2],train.iloc[i][3],train.iloc[i][4]])

			if x>0:
				data=np.vstack((data,final_img))
				target=np.vstack((target,labels))
			else:
				data=final_img
				target=labels


			x+=1
			i+=1
		yield data,target


def show(data,target):
	for x in range(data.shape[0]):
		img=np.reshape(data[x,:,:,:3],[480,640,3])
		labels=target[x]
		print(labels)
		cv2.rectangle(img,(70,20),(619,394),(0,255,0),15)
		cv2.imshow('img',img)
		cv2.waitKey(0)
		cv2.destroyAllWindows()

