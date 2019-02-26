

import tensorflow as tf
import numpy as np
import pandas as pd 
import os
import take_input as input_
import cv2 as cv

FLAGS = tf.app.flags.FLAGS
#tf.app.flags.DEFINE_integer('rows', 20, "num rows")
tf.app.flags.DEFINE_string('file_name','test.csv','name of csv file')

def predict(test_csv):
	tf.reset_default_graph()
	sess=tf.Session()

	new_saver = tf.train.import_meta_graph(os.path.join(os.getcwd(),'saved_model','var.ckpt.meta'))
	new_saver.restore(sess,os.path.join(os.getcwd(),'saved_model','var.ckpt'))
	print('graph restored')
	ops=sess.graph.get_operations()
	#for x in ops:
	#	print(x.name)


	sess_input=sess.graph.get_tensor_by_name('placeholder/image:0')
	sess_out=sess.graph.get_tensor_by_name('fc2layer/BiasAdd:0')
	sess_keep_prob=sess.graph.get_tensor_by_name('placeholder/keep_prob:0')
	sess_training=sess.graph.get_tensor_by_name('placeholder/is_training:0') 


	batch_size=16
	tot_size=test_csv.shape[0]
	num_batches=int(np.ceil(tot_size/batch_size))
	for x in range(num_batches):
		start=x*batch_size
		end=(x+1)*batch_size
		if end>tot_size:
			end=tot_size
		test_images=retrive_image(test_csv.iloc[start:end])
		out=sess.run(sess_out,feed_dict={sess_input:test_images,sess_keep_prob:1.0,sess_training:False})    
		if x==0:
			final_out=out
		else:
			final_out=np.concatenate((final_out,out),axis=0)

		print('tested for {} out of {}'.format(final_out.shape[0],tot_size))
	sess.close()

	return final_out


def retrive_image(addrs):
	for x in range(addrs.shape[0]):
		img_addr=os.path.join(os.getcwd(),'images',addrs.iloc[x,0])
		img=cv.imread(img_addr)
		img=cv.medianBlur(img,5)
		img=cv.medianBlur(img,5)
		img=cv.medianBlur(img,5)
		mask=cv.Canny(img,100,100)
		mask=cv.dilate(mask,(5,5),iterations = 10)
		mask=cv.erode(mask,(5,5),iterations = 15)
		mask=np.reshape(mask,[480,640,-1])
		final_img=np.concatenate((img,mask),axis=2)
		final_img=np.reshape(final_img,[1,480,640,4])
		if x==0:
			img_stack=final_img
		else:
			img_stack=np.vstack((img_stack,final_img))

	return img_stack


def display_test(addr):
	test=pd.read_csv(addr).sample(frac=1)
	print('test shape ',test.shape)
	data=retrive_image(test.iloc[:5,:])
	data=data[:,:,:,:3]
	print('data shape ',data.shape)
	labels=np.array(test.iloc[:10,1:])
	print(labels)
	input_.show(data,labels)


name_file=FLAGS.file_name
test_csv=pd.read_csv(name_file)
print(test_csv.shape)


if os.path.isfile(os.path.join(os.getcwd(),'preds.npy'))==False:
	preds=predict(test_csv)
	np.save('preds.npy',preds)

preds=np.load('preds.npy')
print(preds.shape)

preds[:,[0,2]]=np.clip(preds[:,[0,2]],0,480)
preds[:,[1,3]]=np.clip(preds[:,[1,3]],0,640)
preds=preds.astype(np.int16)

labels=pd.DataFrame()
labels['image_name']=test_csv['image_name']
labels['x1']=preds[:,0]
labels['x2']=preds[:,1]
labels['y1']=preds[:,2]
labels['y2']=preds[:,3]

labels.to_csv('output1.csv',index=False)

display_test('output1.csv')




