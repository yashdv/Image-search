/*
  Commands and summary of functions:

  gist-classifier image <image-path>

    Given <image-path>, shrinks image to 128x128 size and prints its
    GIST descriptor to standard output.

  gist-classifier calc_features <image-list> <savexml>
  
    Given <image-list>, a text file with the path of an image file on
    every line, for each image file, extract the GIST descriptor
    (960-element vector) and add the vector to a training matrix.
    Then save the training matrix as the XML file specified by
    <savexml>.

  gist-classifier train_svm <positive.xml> <negative.xml> <model.xml>

    positive.xml contains an openCV training matrix of GIST
    descriptors of images in the "positive" class (class 1).

    negative.xml contains an openCV training matrix of GIST
    descriptors of images in the "negative" class (class 2).

    Train an SVM with the positive and negative training matrices, and
    save the resulting openCV SVM algorithm in <model.xml>

  gist-classifier test_svm xml <positive.xml> <negative.xml> <model.xml> <optional-pred.xml>
  
    Given a trained SVM (stored in <model.xml>), test_svm tests the
    SVM by running the samples stored in <positive.xml> and
    <negative.xml> testing matrices.  The number of true positives,
    true negatives, false positives, and false negatives are printed
    to standard out.    

  gist-classifier video <video-path> <model1.xml> <model2.xml>
  
    Two trained SVMs are stored in <model1.xml> and <model2.xml>.
    Each frame of the video at <video-path> is classified as either
    positive for model1, positive for model2, or negative for both
    model1 and model2.  Each frame is correspondingly labeled as "1",
    "2", or "-" in the upper left corner.

  gist-classifier camera <model1.xml> <model2.xml>

    Same as gist-classifier video, except with live camera capture
    instead of recorded video file.
 */

#include <iostream>
#include <fstream>
#include "highgui.h"
#include "cv.h"
#include "ml.h"
#include <cstdio>
#include <string>
#include <cstring>
#include <cmath>
#include "gist-classifier.hpp"
extern "C" {
  #include "gist.h"
  #include "standalone_image.h"
}
using namespace std;

const static int GIST_SIZE = 128;
const static int feature_vector_length = 960;
const static int nblocks = 4;
const static int n_scale = 3;
const static int orientations_per_scale[50]={8,8,4};
const static int pos_label = 1;
const static int neg_label = 2;
static image_list_t *Gabor;

/* Displays image in window, which closes upon user keypress. */
void flash_image(IplImage* img, const char* window_name) {
  cvNamedWindow(window_name, CV_WINDOW_AUTOSIZE);
  cvShowImage(window_name, img);
  char c = cvWaitKey(0);
  cvDestroyWindow(window_name);
}

/* Convert OpenCV IplImage into LEAR's desired color_image_t format. */
/* Direct access using a c++ wrapper using code found under topic
   "Simple and efficient access" at link:*/
/* http://www.cs.iit.edu/~agam/cs512/lect-notes/opencv-intro/opencv-intro.html#SECTION00053000000000000000*/
void convert_IplImage_to_LEAR(IplImage* src, color_image_t* dst) {
  assert(src->width == GIST_SIZE && src->height == GIST_SIZE);
  assert(src->depth == IPL_DEPTH_8U);
  RgbImage imgA(src);
  int x, y, i = 0;
  for (y = 0; y < GIST_SIZE; y++) {
    for (x = 0; x < GIST_SIZE; x++) {
      dst->c1[i] = imgA[y][x].r;
      dst->c2[i] = imgA[y][x].g;
      dst->c3[i] = imgA[y][x].b;
      i++;
    }
  }
  assert(i == (GIST_SIZE * GIST_SIZE));
}

/*
 * Computes the GIST descriptor for a copy of img resized to 128x128
 * (GIST_SIZE x GIST_SIZE).
 */
float* my_compute_gist(IplImage* img, IplImage* rsz) {
  /* Resize image to 128x128 before calculating GIST descriptor. */
  assert(img);
  assert(rsz);
  cvResize(img, rsz, CV_INTER_LINEAR);
  /* Lear's compute_gist takes a ppm and computes gist descriptor for it. */
  color_image_t *lear = color_image_new(GIST_SIZE, GIST_SIZE);
  assert(lear);
  convert_IplImage_to_LEAR(rsz, lear);
  assert(Gabor);
  float *desc = color_gist_scaletab_gabor(lear, nblocks, n_scale, orientations_per_scale, Gabor);
  /* Cleanup. */
  color_image_delete(lear);
  return desc;
} 

/* Counts the number non-empty lines in a file. */
int number_of_lines(const char* filename) {
  ifstream file(filename);
  assert(file);
  int n = 0;
  string line;
  while (getline(file, line)) {
    /* Ignore empty lines. */
    if (strlen(line.c_str()) != 0) {
      n++;
    }
  }
  return n;
}

/*
 * For each image file named in imagelist_filename, extract the GIST
 * descriptor (960-element vector) and add the vector into training
 * matrix.  If savexml is provided, store the training matrix in
 * XML-form to file specified by savexml.  This function assumes that
 * the images in imagelist_filename are all "positive" or all
 * "negative" samples.  If you want to mix "positive" and "negative"
 * samples, you'll have to store the image labels in your own matrix
 * and xml-file; this function only calculates the features (GIST
 * descriptors).
 * 
 * imagelist_file : a list of image filenames. path/foo.jpg \n
 * path/bar.jpg, etc.
 * 
 * savexml : file to save the OpenCV training matrix.
 */
CvMat* feature_vectors_img128x128(const char* imagelist_filename,
				  char* savexml = NULL) {
  int number_samples = number_of_lines(imagelist_filename);
  CvMat* training = cvCreateMat(number_samples, feature_vector_length, CV_32FC1);
  CvMat row;
  int i = 0, row_index = 0;
  ifstream imagelist_file(imagelist_filename);
  assert(imagelist_file);
  string filename;
  IplImage *img, *gist_img;
  float *desc;
  
  printf("Beginning to extract GIST descriptors from %s images\n", imagelist_filename);
  while (getline(imagelist_file, filename)) {
    /* Ignore empty lines. */
    if (strlen(filename.c_str()) == 0) {
      continue;
    }
    img = cvLoadImage(filename.c_str());
    if (!img) {
      cout << "Error opening image file named: " << filename << "\n";
      assert(img);
    }
    gist_img = cvCreateImage(cvSize(GIST_SIZE, GIST_SIZE), IPL_DEPTH_8U, 3);
    desc = my_compute_gist(img, gist_img);
    /* Save descriptor in training matrix. */
    assert(row_index < number_samples);
    cvGetRow(training, &row, row_index);
    for (i = 0; i < feature_vector_length; i++) {
      cvSetReal1D(&row, i, desc[i]);
    }
    row_index++;

    /* Clean up descriptor. */
    free(desc);
  }
  assert(row_index == number_samples);

  if (savexml != NULL) {
    cvSave(savexml, training);
  }
  /* Clean up. */
  imagelist_file.close();
  return training;
}

/* http://smsoftdev-solutions.blogspot.com/2009/10/object-detection-using-opencv-iii.html */
/* This function trains a linear support vector
machine for object classification. The synopsis is
as follows :

pos_mat : pointer to CvMat containing hog feature
          vectors for positive samples. This may be
          NULL if the feature vectors are to be read
          from an xml file

neg_mat : pointer to CvMat containing hog feature
          vectors for negative samples. This may be
          NULL if the feature vectors are to be read
          from an xml file

savexml : The name of the xml file to which the learnt
          svm model should be saved

pos_file: The name of the xml file from which feature
          vectors for positive samples are to be read.
          It may be NULL if feature vectors are passed
          as pos_mat

neg_file: The name of the xml file from which feature
          vectors for negative samples are to be read.
          It may be NULL if feature vectors are passed
          as neg_mat*/


void trainSVM(CvMat* pos_mat, CvMat* neg_mat, char *savexml,
              char *pos_file = NULL, char *neg_file = NULL) {
  
  /* Read the feature vectors for positive samples */
  if (pos_file != NULL) {
      pos_mat = (CvMat*) cvLoad(pos_file);
      printf("positive loaded\n");
  }
  assert(pos_mat);

  /* Read the feature vectors for negative samples */
  if (neg_file != NULL) {
      neg_mat = (CvMat*) cvLoad(neg_file);
      printf("negative loaded\n");
  }
  assert(neg_mat);

  int n_positive, n_negative;
  n_positive = pos_mat->rows;
  n_negative = neg_mat->rows;
  int feature_vector_length = pos_mat->cols;
  int total_samples;
  total_samples = n_positive + n_negative;

  CvMat* trainData = cvCreateMat(total_samples,
				 feature_vector_length, CV_32FC1);

  CvMat* trainClasses = cvCreateMat(total_samples,
				    1, CV_32FC1 );

  CvMat trainData1, trainData2, trainClasses1,
    trainClasses2;

  printf("Number of positive Samples : %d\n",
         pos_mat->rows);
 
  /*Copy the positive feature vectors to training
    data*/

  cvGetRows(trainData, &trainData1, 0, n_positive);
  cvCopy(pos_mat, &trainData1);
  cvReleaseMat(&pos_mat);

  /*Copy the negative feature vectors to training
    data*/

  cvGetRows(trainData, &trainData2, n_positive,
	    total_samples);

  cvCopy(neg_mat, &trainData2);
  cvReleaseMat(&neg_mat);

  printf("Number of negative Samples : %d\n",
         trainData2.rows);

  /*Form the training classes for positive and
    negative samples. Positive samples belong to class
    1 and negative samples belong to class 2 */

  cvGetRows(trainClasses, &trainClasses1, 0, n_positive);
  cvSet(&trainClasses1, cvScalar(pos_label));
 
  cvGetRows(trainClasses, &trainClasses2, n_positive,
	    total_samples);

  cvSet(&trainClasses2, cvScalar(neg_label));
 
 
  /* Train a linear support vector machine to learn from
     the training data. The parameters may played and
     experimented with to see their effects*/
 
  CvSVM svm(trainData, trainClasses, 0, 0,
	    CvSVMParams(CvSVM::C_SVC, CvSVM::LINEAR, 0, 0, 0, 2,
			0, 0, 0, cvTermCriteria(CV_TERMCRIT_EPS,0, 0.01)));

  printf("SVM Training Complete!!\n");
 
  /*Save the learnt model*/
 
  if (savexml != NULL) {
    svm.save(savexml);
  }
  cvReleaseMat(&trainClasses);
  cvReleaseMat(&trainData);

}

/*
 * Given a trained SVM (stored in model_xml), testSVM tests the SVM by
 * running the samples stored in pos_mat and neg_mat and counting the
 * correct and incorrect predictions.  The number of true positives,
 * true negatives, false positives, and false negatives are printed to
 * standard out at the end.
 */
void testSVM(CvMat* pos_mat, CvMat* neg_mat, char *model_xml,
	     char *pos_file = NULL, char *neg_file = NULL, char *pred_file = NULL) {
  /* Read the feature vectors for positive samples */
  if (pos_file != NULL) {
    pos_mat = (CvMat*) cvLoad(pos_file);
    printf("positive loaded.\n");
  }
  assert(pos_mat);

  /* Read the feature vectors for negative samples */
  if (neg_file != NULL) {
    neg_mat = (CvMat*) cvLoad(neg_file);
    printf("negative loaded.\n");
  }
  assert(neg_mat);

  assert(model_xml != NULL);
  CvSVM svm;
  svm.load(model_xml);
  printf("SVM model loaded.\n");
  
  CvMat row;
  float prediction;
  int i, true_pos = 0, true_neg = 0, false_pos = 0, false_neg = 0;
  float error = 0.5;
  for (i = 0; i < pos_mat->rows; i++) {
    cvGetRow(pos_mat, &row, i);
    prediction = svm.predict(&row);
    if (abs(prediction - pos_label) < error) {
      true_pos++;
    } else {
      false_neg++;
    }
  }
  for (i = 0; i < neg_mat->rows; i++) {
    cvGetRow(neg_mat, &row, i);
    prediction = svm.predict(&row);
    if (abs(prediction - neg_label) < error) {
      true_neg++;
    } else {
      false_pos++;
    }
  }
  printf("true_pos = %d, true_neg = %d, false_pos = %d, false_neg = %d\n",
	 true_pos, true_neg, false_pos, false_neg);
  
}


/*
 * Given a trained SVM (or two), for_capture takes frames from the
 * capture's video stream and classifies the frame with the SVM.o
 *
 * model_xml : an XML file where an OpenCV SVM classifier is stored.
 * The SVM takes GIST descriptors as input and outputs a
 * positive/negative classification.
 * 
 * capture : a camera or video capture where frames/images are drawn
 * from.
 */
int for_capture(CvCapture* capture, char* model_xml1, char* model_xml2=NULL) {
  CvSVM svm1, svm2;
  printf("Loading SVM1 model.\n");
  svm1.load(model_xml1);
  printf("SVM1 model loaded.\n");

  if (model_xml2) {
    printf("Loading SVM2 model.\n");
    svm2.load(model_xml2);
    printf("SVM2 model loaded.\n");
  }

  /* Init video read. */
  IplImage* frame = cvQueryFrame(capture);
  assert(frame);
  double mspf = 33; /* Default milliseconds per frame. */
  double fps = cvGetCaptureProperty(capture, CV_CAP_PROP_FPS);
  if (fps != 0) mspf = 1000/fps; /* Set milliseconds per frame. */
  IplImage* gist_frame = cvCreateImage(cvSize(GIST_SIZE, GIST_SIZE), IPL_DEPTH_8U, 3);
  const char* gist_video = "gist_video";
  cvNamedWindow(gist_video, CV_WINDOW_AUTOSIZE);
  float *desc;
  CvMat *row = cvCreateMat(1, feature_vector_length, CV_32FC1);
  char c;
  int i;
  float pred1, pred2;
  float error = 0.5;

  /* For text on video. */
  CvFont font;
  double hScale=1.0;
  double vScale=1.0;
  int    lineWidth=1;
  string label = "";
  cvInitFont(&font,CV_FONT_HERSHEY_SIMPLEX|CV_FONT_ITALIC, hScale,vScale,0,lineWidth);
  /* Play video. */
  while(1) {
    frame = cvQueryFrame(capture);
    if (!frame) break;
    /* Some transformation. */
    desc = my_compute_gist(frame, gist_frame);
    for (i = 0; i < feature_vector_length; i++) {
      cvSetReal1D(row, i, desc[i]);
    }
    free(desc);
    pred1 = svm1.predict(row);
    label = "-"; /* Negative */
    if (abs(pred1 - pos_label) < error) {
      label = "1"; /* Positive for Model 1. */
    } else if (model_xml2) {
      pred2 = svm2.predict(row);
      if (abs(pred2 - pos_label) < error) {
	label = "2"; /* Positive for Model 2. */
      }
    }
    cvPutText (gist_frame, label.c_str(), cvPoint(0,30), &font, cvScalar(255,255,0));
    cvShowImage(gist_video, gist_frame);
    c = cvWaitKey(mspf);
    if (c == 27) break; /* 27 is keycode for ESC. */
  }
  /* Clean up. */
  cvReleaseCapture(&capture);
  cvDestroyWindow(gist_video);
  return 0;
}

/* See for_capture. */
int for_video_capture(int argc, char** argv) {
  if (argc < 4) {
    printf("Usage: gist-classifier video <video-path> <model_xml> <optional model_xml2>\n");
    return 0;
  }
  char* model_xml = argv[3];
  char* model_xml2= NULL;
  if (argc > 4) {
    model_xml2 = argv[4];
  }
  CvCapture* capture;
  printf("Open video file.\n");
  capture = cvCreateFileCapture(argv[2]);
  if (!capture) {
    printf("Error opening video file %s.\n", argv[2]);
    return 0;
  }
  return for_capture(capture, model_xml, model_xml2);
}

/* See for_capture. */
int for_camera_capture(int argc, char** argv) {
  if (argc < 3) {
    printf("Usage: gist-classifier camera <model_xml> <optional model_xml2>\n");
    return 0;
  }
  char* model_xml = argv[2];
  char* model_xml2= NULL;
  if (argc > 3) {
    model_xml2 = argv[3];
  }
  CvCapture* capture;
  printf("Open camera capture.\n");
  capture = cvCreateCameraCapture(0);
  if (!capture) {
    printf("Error opening camera capture.\n");
    return 0;
  }
  return for_capture(capture, model_xml, model_xml2);
}

/*
 * Compute the GIST descriptor of the image and print the 960
 * descriptor values to standard out.
 */
int forImage(int argc, char** argv) {
  if (argc < 3) {
    printf("Usage: gist-classifier image <image-path>\n");
    return 0;
  }
  IplImage* img = cvLoadImage(argv[2]);
  if (!img) {
    printf("Unable to load image: %s\n", argv[2]);
    return 0;
  }
  /* Do something with image.  Calculate GIST descriptor. */
  IplImage* gist_img = cvCreateImage(cvSize(GIST_SIZE, GIST_SIZE), IPL_DEPTH_8U, 3);
  float* desc = my_compute_gist(img, gist_img);
  /* Should contain feature_vector_length (960) float values */
  for (int i = 0; i < feature_vector_length; i++) {
    printf("%d: %f\n", i, desc[i]);
  }
  free(desc);
  flash_image(img, "Original");
  cvReleaseImage(&img);
  return 0;
}

int for_calc_features(int argc, char** argv) {
  if (argc != 4) {
    printf("Usage: gist-classifier calc_features <image-list> <savexml>\n");
    return 0;
  }
  string calc_features("calc_features");
  char* option = argv[1];
  assert(calc_features.compare(option) == 0);
  char* imagelist_filename = argv[2];
  char* savexml = argv[3];
  assert(savexml);
  
  feature_vectors_img128x128(imagelist_filename, /*number_samples,*/ savexml);
  return 0;
}

int for_train_svm(int argc, char** argv) {
  if (argc < 6) {
    printf("Usage: gist-classifier train_svm xml <positive.xml> <negative.xml> <model.xml>\n");
    return 0;
  }
  string train("train_svm");
  char* option = argv[1];
  assert(train.compare(option) == 0);
  string xml("xml");
  char* data_option = argv[2];
  assert(xml.compare(data_option) == 0);
  char* pos_xml = argv[3];
  char* neg_xml = argv[4];
  char* model_xml = argv[5];

  trainSVM(NULL/*pos_mat*/, NULL/*neg_mat*/, model_xml, pos_xml, neg_xml);
  return 0;
}

int for_test_svm(int argc, char** argv) {
  if (argc < 6) {
    printf("Usage: gist-classifier test_svm xml <positive.xml> <negative.xml> <model.xml> <optional-pred.xml>\n");
    return 0;
  }
  string test("test_svm");
  char* option = argv[1];
  assert(test.compare(option) == 0);
  string xml("xml");
  char* data_option = argv[2];
  assert(xml.compare(data_option) == 0);
  char* pos_xml = argv[3];
  char* neg_xml = argv[4];
  char* model_xml = argv[5];
  testSVM(NULL/*pos_mat*/, NULL/*neg_mat*/,
	  model_xml, pos_xml, neg_xml, (argc == 7) ? argv[6] : NULL);
  return 0;
}

int main(int argc, char** argv) {
  if (argc < 2) {
    printf("Usage: gist-classifier <option> <optional image or video path>\n");
    return 0;
  }
  Gabor = create_gabor(n_scale, orientations_per_scale, GIST_SIZE, GIST_SIZE);
  string video("video");
  string image("image");
  string camera("camera");
  string calc("calc_features");
  string train("train_svm");
  string test("test_svm");
  char* option = argv[1];
  if (video.compare(option) == 0) {
    return for_video_capture(argc, argv);
  } else if (camera.compare(option) == 0) {
    return for_camera_capture(argc, argv);
  } else if (image.compare(option) == 0) {
    return forImage(argc, argv);
  } else if (calc.compare(option) == 0){
    return for_calc_features(argc, argv);
  } else if (train.compare(option) == 0) {
    return for_train_svm(argc, argv);
  } else if (test.compare(option) == 0) {
    return for_test_svm(argc, argv);
  } else {
    printf("Unknown option %s\n", option);
    return 0;
  }
}
