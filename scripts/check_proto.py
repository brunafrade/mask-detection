import caffe

try:
    caffe.Net('models/ssd_face/ssd_face_train.prototxt', 
              caffe.TRAIN)
    caffe.Net('models/ssd_face/ssd_face_test.prototxt', 
               
              caffe.TEST)
    caffe.Net('models/ssd_face/ssd_face_deploy.prototxt', 
               
              caffe.TEST)
    caffe.Net('models/ssd_face/ssd_face_deploy_bn.prototxt', 
               
              caffe.TEST)
    print('Model check COMPLETE')
except Exception as e:
    print(repr(e))
    print('Model check FAILED')
    
