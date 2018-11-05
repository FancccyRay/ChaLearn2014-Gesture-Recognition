#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri May 25 21:34:42 2018

@author: fancy
"""

"""
Step 7: compute the state matrix to get predicted labels,test the classification quality using 
viterbi decoding and  compute the score of predicted data using Jaccard Index
"""
import os
import sys
import cPickle
import cv2
import numpy
import scipy.io as sio  
import csv
import shutil

from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import *

from ui_show_result_video import Ui_MainWindow,Ui_GestureTabelCheck

from classes import GestureSample
from functions.preproc_functions import *
from functions.test_functions import *

## Load Prior and transitional Matrix
dic=sio.loadmat('Prior_Transition_matrix_5states.mat')
Transition_matrix = dic['Transition_matrix']
Prior = dic['Prior']

#classes of gesture
gesture_name = ['vattene', 'vieniqui','perfetto','furbo','cheduepalle',
        'chevuoi','daccordo','seipazzo','combinato','freganiente',
        'ok','cosatifarei','basta','prendere','noncenepiu','fame',
        'tantotempo','buonissimo','messidaccordo','sonostufo']


class GesturesRecognization(QWidget,Ui_MainWindow):
    def __init__(self):
        super(GesturesRecognization, self).__init__()
        self.cap = cv2.VideoCapture()
        self.setupUi(self)
        self.first_load_flag = False
        self.timer_camera = QtCore.QTimer()
        self.slot_init()
        self.pause_flag = False
        self.samplePath = ''
        self.video_file = ''
        self.seqID = ''
        self.pred_path = ''     
    
    def slot_init(self):
        self.LoadButton.clicked.connect(self.loadfile)
        self.StartButton.clicked.connect(self.startvideo)
        self.PauseButton.clicked.connect(self.pausevideo)
        self.timer_camera.timeout.connect(self.videoShow)
        
        self.CloseButton.clicked.connect(self.close)
        self.CloseButton.clicked.connect(self.table_close)
        
        self.CheckButton.clicked.connect(self.table_show)
        self.CheckButton.clicked.connect(self.checkdata)
        
    
    def loadfile(self):
        if self.first_load_flag:
            self.ImageShowLabel.clear()
            self.cap.release()
            shutil.rmtree(self.samplePath)
            
        self.LoadingLabel.setText('Loading...........')
        video_file, filetype = QFileDialog.getOpenFileName(self,  
        'choose a video', '/home/fancy/Desktop/test_videos/',"Image files (*.zip)")
        if not os.path.exists(video_file):
            self.LoadingLabel.setText('')
            msg = QtWidgets.QMessageBox.warning(self,u"Warning",u"Please check the file again,this zip file is unavailabel",
                                    buttons=QtWidgets.QMessageBox.Ok,defaultButton=QtWidgets.QMessageBox.Ok)
            self.LoadingLabel.setText('Please choose a zip file to load')
        else:
            
            filename = os.path.split(video_file)[1]
            matrix_file = '/home/fancy/Desktop/matrix_files/' + filename
            self.pred_path = '/home/fancy/Desktop/test_videos_pred' 
            self.video_file = video_file
            self.VideoLabel.setText(filename)
            
            self.predLabel(self.video_file,matrix_file, self.pred_path,filename)
            self.scoreGesture(self.pred_path ,self.video_file, filename, begin_add=0, end_add=0)
            self.LoadingLabel.setText('You can start watching the video now ')
            self.first_load_flag = True
            
        
        
    def predLabel(self, video_file, matrix_file, pred_file, filename):
#        time_start = time.time() 
        print("\t Processing file " + filename)     
        sample_video = GestureSample(video_file)
        observ_likelihood = cPickle.load(open(matrix_file,"rb"))
#        print 'fininsh Loadinging obs_likelihodd'
#        print observ_likelihood.shape
            
        log_observ_likelihood = log(observ_likelihood.T + numpy.finfo(numpy.float32).eps)
        log_observ_likelihood[-1, 0:5] = 0
        log_observ_likelihood[-1, -5:] = 0
            
        #viterbi decoding
        [path, predecessor_state_index, global_score] = viterbi_path_log(log(Prior), 
        log(Transition_matrix), log_observ_likelihood)
        [pred_label, begin_frame, end_frame, Individual_score, frame_length] = viterbi_colab_states(path, 
        global_score, state_no = 5, threshold=-5, mini_frame=15)
        #heuristically we need to add 1 more frame here
        begin_frame += 1 
        end_frame +=5 # because we cut 4 frames as a cuboid so we need add extra 4 frames 
            
        #plotting
        gesturesList=sample_video.getGestures()
        import matplotlib.pyplot as plt
        STATE_NO = 5
        im  = imdisplay(global_score)
            
        plt.plot(range(global_score.shape[-1]), path, color='#39FF14',linewidth=2.0)
        plt.xlim((0, global_score.shape[-1]))
        plt.ylim((101,0))
        plt.xlabel('Frames')
        plt.ylabel('HMM states')
        plt.title('Multi_model(DBN+3DCNN)')

        # plot ground truth
        for gesture in gesturesList:
            gestureID,startFrame,endFrame=gesture
            frames_count = numpy.array(range(startFrame, endFrame+1))
            pred_label_temp = ((gestureID-1) *STATE_NO +2) * numpy.ones(len(frames_count))
            plt.plot(frames_count, pred_label_temp, color='r', linewidth=5.0)
          
        # plot clean path
        for i in range(len(begin_frame)):
            rames_count = numpy.array(range(begin_frame[i], end_frame[i]+1))
            pred_label_temp = ((pred_label[i]-1) *STATE_NO +2) * numpy.ones(len(frames_count))
            plt.plot(frames_count, pred_label_temp, color='#FFFF33', linewidth=2.0)
        plt.show()  
        
        pred=[]
        for i in range(len(begin_frame)):
            pred.append([ pred_label[i], begin_frame[i], end_frame[i]] )
        sample_video.exportPredictions(pred,pred_file)
#        print"viterbi me used: %d sec"%int(time.time()-time_start)
#        del sample_video
       
        
    def scoreGesture(self,prediction_dir,truth_dir, filename,begin_add=0, end_add=0):  
#        time_start = time.time() 
        file_path = os.path.split(truth_dir)[0]
        self.seqID=os.path.splitext(filename)[0]
        self.samplePath = file_path + os.path.sep + self.seqID
        os.chdir(file_path)
        if not os.path.exists(truth_dir):
            raise Exception("Sample path does not exist: " + filename)
        if os.path.isdir(self.samplePath) == False:
            zipFile=zipfile.ZipFile(truth_dir,"r")
            zipFile.extractall(self.samplePath)
        else:
            print 'zipfile is already decompressed'
            
#        rgbFile = os.path.join(self.samplePath, self.seqID + "_color.mp4")
        labelsFile = os.path.join(self.samplePath, self.seqID + "_labels.csv")
        dataFile = os.path.join(self.samplePath, self.seqID + "_data.csv")
        predFile = os.path.join(prediction_dir, self.seqID + "_prediction.csv")
        # Get the number of frames for this sample
        with open(dataFile, 'rb') as csvfile:
            filereader = csv.reader(csvfile, delimiter=',')
            for row in filereader:
                numFrames=int(row[0])
#                print numFrames
            del filereader
        
        score = self.gesture_overlap_csv(labelsFile, predFile,self.seqID,numFrames, begin_add, end_add)
        print "Sample ID: %s, score %f" %(self.seqID,score)
#        print"scroe computing time used: %d sec"%int(time.time()-time_start)      
#        shutil.rmtree(self.samplePath)
        

    def gesture_overlap_csv(self,csvpathgt, csvpathpred,seqID, seqlenght, begin_add, end_add):
        """ Evaluate this sample agains the ground truth file """
        maxGestures=20

        # Get the list of gestures from the ground truth and frame activation
        gtGestures = []
        binvec_gt = numpy.zeros((maxGestures, seqlenght))
        with open(csvpathgt, 'rb') as csvfilegt:
            csvgt = csv.reader(csvfilegt)
            for row in csvgt:
                binvec_gt[int(row[0])-1, int(row[1])-1:int(row[2])-1] = 1
                gtGestures.append(int(row[0]))
        # Get the list of gestures from prediction and frame activation
        predGestures = []
        binvec_pred = numpy.zeros((maxGestures, seqlenght))
        with open(csvpathpred, 'rb') as csvfilepred:
            csvpred = csv.reader(csvfilepred)
            for row in csvpred:
                binvec_pred[int(row[0])-1, int(row[1])-1+begin_add:int(row[2])-1+end_add] = 1
                predGestures.append(int(row[0]))             
        #################################################
        ##show data in table
        for row in gtGestures:
            i = 0
            for temp in gtGestures:
                if row == temp:
                    i += 1
            newItem = QTableWidgetItem(str(i))
            self.GesturesTable.setItem(row,0,newItem)
            
        for row in predGestures:
            j = 0
            for temp in predGestures:
                if row ==temp:
                    j += 1
            newItem = QTableWidgetItem(str(j))
            self.GesturesTable.setItem(row,1,newItem)

         
        
        # Get the list of gestures without repetitions for ground truth and predicton
        if len(numpy.unique(gtGestures)) != len(gtGestures):
            print "not unique!"
        gtGestures_unique = numpy.unique(gtGestures)
        predGestures_unique = numpy.unique(predGestures)
#        print len(gtGestures), len(predGestures)
        # Find false positives
        falsePos=numpy.setdiff1d(numpy.union1d(gtGestures_unique,predGestures_unique),gtGestures_unique)

        # Get overlaps for each gesture
        #not according to the gtGesture order
        overlaps = []
        for idx in gtGestures_unique:
            intersec = sum(binvec_gt[idx-1] * binvec_pred[idx-1])
            aux = binvec_gt[idx-1] + binvec_pred[idx-1]
            union = sum(aux > 0)
            overlap_prob = intersec/union
            ###########show overlap in tabel
            newItem = QTableWidgetItem(str(overlap_prob))
            self.GesturesTable.setItem(idx,2,newItem)
            overlaps.append(overlap_prob)
            
#        print overlaps
#        """ Export the given overlap probabilities to the correct file in the given path """
#        """save overlaps data"""
        output_filename = os.path.join(self.pred_path, self.seqID + '_overlaps.csv')
        output_file = open(output_filename, 'wb')
        for row in overlaps:
            output_file.write(repr(row) + "\n")
        output_file.close()
#        print len(overlaps)+len(falsePos)
        # Use real gestures and false positive gestures to calculate the final score
        score = sum(overlaps)/(len(overlaps)+len(falsePos))
        ####show score in label
        self.ScoreLabel.setText(str(score))
        return score
    
    
    def videoShow(self):
        ret, frame = self.cap.read()
        
        if ret:    
            frame = cv2.resize(frame, (640, 480))
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            if (self.current_frame >= self.gt_begin_frame) and ( self.current_frame <= self.gt_end_frame):
                gt_name = gesture_name[int(self.gtGestures_read[self.gt_gesture_count][0])-1]
                cv2.putText(frame,'Truth:'+ str(self.gt_gesture_count+1) ,(10,100),cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),4)
                cv2.putText(frame,gt_name,(10,240),cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),4)
                if self.current_frame == self.gt_end_frame:
                    self.gt_gesture_count += 1
                    self.gt_begin_frame = int(self.gtGestures_read[self.gt_gesture_count][1])
                    self.gt_end_frame = int(self.gtGestures_read[self.gt_gesture_count][2])
                        
            if (self.current_frame >= self.pred_begin_frame) and ( self.current_frame <= self.pred_end_frame):
                pred_name = gesture_name[int(self.predGestures_read[self.pred_gesture_count][0])-1]
                cv2.putText(frame,'Predicted:'+str(self.pred_gesture_count+1),(400,100),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),4)
                cv2.putText(frame,pred_name,(450,240),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),4)
                if self.current_frame == self.pred_end_frame:
                    self.pred_gesture_count += 1
                    self.pred_begin_frame = int(self.predGestures_read[self.pred_gesture_count][1])
                    self.pred_end_frame = int(self.predGestures_read[self.pred_gesture_count][2]) 
                        
            showImage = QtGui.QImage(frame.data, frame.shape[1], frame.shape[0], QtGui.QImage.Format_RGB888)
            self.ImageShowLabel.setPixmap(QtGui.QPixmap.fromImage(showImage))
            
            self.current_frame += 1

        else:
            self.cap.release()
            self.timer_camera.stop()
        
                
        
    def startvideo(self):
        self.LoadingLabel.setText("")
        if self.first_load_flag:
            self.ImageShowLabel.setText("")
            self.cap.release()
#            cv2.destroyWindow("test")
        
        self.current_frame = 0
                   
        self.gtGestures_read,self.predGestures_read,overlaps_no_use,= self.LoadFileData()
        
        rgbFile = os.path.join(self.samplePath, self.seqID + "_color.mp4")
        if not os.path.exists(rgbFile):
            raise Exception("Invalid sample file. RGB data is not available")
        self.cap = cv2.VideoCapture(rgbFile)
        while not self.cap.isOpened():
            self.cap = cv2.VideoCapture(rgbFile)
            cv2.waitKey(100)
                
        self.gt_gesture_count = 0
        self.gt_begin_frame = int(self.gtGestures_read[self.gt_gesture_count][1])
        self.gt_end_frame = int(self.gtGestures_read[self.gt_gesture_count][2])
        
        self.pred_gesture_count = 0
        self.pred_begin_frame = int(self.predGestures_read[self.pred_gesture_count][1])
        self.pred_end_frame = int(self.predGestures_read[self.pred_gesture_count][2])
            
        self.timer_camera.start(30)
      

            
    def LoadFileData(self):
        if os.path.isdir(self.samplePath) == False:
            zipFile=zipfile.ZipFile(self.video_file,"r")
            zipFile.extractall(self.samplePath)
        else:
            print 'zipfile is already decompressed'
        gtGestures_read = []
        labelsFile = os.path.join(self.samplePath, self.seqID + "_labels.csv")
        with open(labelsFile, 'rb') as csvfilepred:
            csvpred = csv.reader(csvfilepred)
            for row in csvpred:
                gtGestures_read.append(row)
        predGestures_read = []
        predFile = os.path.join(self.pred_path, self.seqID + "_prediction.csv")
        with open(predFile, 'rb') as csvfilepred:
            csvpred = csv.reader(csvfilepred)
            for row in csvpred:
                predGestures_read.append(row) 
        overlaps = []
        overlapFile = os.path.join(self.pred_path, self.seqID + '_overlaps.csv')
        with open(overlapFile,'rb') as csvfilepred:
            csvfile = csv.reader(csvfilepred)
            for row in csvfile:
                overlaps.append(row)
        
        return gtGestures_read , predGestures_read , overlaps
    
    
    def pausevideo(self):
        if self.timer_camera.isActive() == True:
            self.timer_camera.stop()
        else:
            self.timer_camera.start()
        
                    
    def checkdata(self):
        gtGestures_read, predGestures_read , overlaps = self.LoadFileData()
           
        for row in gtGestures_read:
            i = 0
            for temp in gtGestures_read:
                if row[0] == temp[0]:
                    i += 1
            newItem = QTableWidgetItem(str(i))
            self.gt_table.GesturesTable.setItem(int(row[0]),0,newItem)
            
        for row in predGestures_read:
            j = 0
            for temp in predGestures_read:
                if row[0] ==temp[0]:
                    j += 1
            newItem = QTableWidgetItem(str(j))
            self.gt_table.GesturesTable.setItem(int(row[0]),1,newItem)
            
        gtGestures_class = [] 
        for row in gtGestures_read:
            gtGestures_class.append(int(row[0]))    
        gtGestures_class = numpy.unique(gtGestures_class)
        n_overlap = 0
        for row in gtGestures_class:
            newItem = QTableWidgetItem(str(overlaps[n_overlap][0]))
            self.gt_table.GesturesTable.setItem(row,2,newItem)
            n_overlap += 1
#        shutil.rmtree(self.samplePath)
    
    def closeEvent(self, event):
        ok = QtWidgets.QPushButton()
        cancel = QtWidgets.QPushButton()

        msg = QtWidgets.QMessageBox(QtWidgets.QMessageBox.Warning, u"CLOSE", u"Do you really want to close it?")

        msg.addButton(ok  ,QtWidgets.QMessageBox.ActionRole)
        msg.addButton(cancel, QtWidgets.QMessageBox.RejectRole)
        ok.setText(u'Sure')
        cancel.setText(u'Cancel')
        # msg.setDetailedText('sdfsdff')
        if msg.exec_() == QtWidgets.QMessageBox.RejectRole:
            event.ignore()
        else:
            #             self.socket_client.send_command(self.socket_client.current_user_command)
            if self.cap.isOpened():
                self.cap.release()
            if self.timer_camera.isActive():
                self.timer_camera.stop()
            if os.path.isdir(self.samplePath):
                shutil.rmtree(self.samplePath)
            event.accept()
        
         
        
if __name__=="__main__":
    app = QApplication(sys.argv)
    GR = GesturesRecognization()
    GR.show()
    sys.exit(app.exec_())


    


