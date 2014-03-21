#include <iostream>
#include <cstdio>
#include <cstring>
#include <vector>
#include <set>

#include <opencv/highgui.h>
#include <opencv/cv.h>
#include <boost/filesystem.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/nonfree/features2d.hpp>
#include <opencv2/nonfree/nonfree.hpp>
#include <opencv2/legacy/legacy.hpp>

using namespace std;
using namespace boost::filesystem;
using namespace cv;

#define tr(c,i) for(typeof((c).begin()) i = (c).begin(); i != (c).end(); i++)

int num_q = 0;
int succ_q = 0;
int trainFLAG;
int display_flag;

char TRAIN_DATA_PATH[128]  = "training_data/";
char TRAIN_VOCAB_PATH[128] = "training_vocab/";
char EVAL_DATA_PATH[128]   = "testing_data/";
char all_data[128]         = "all_data/";
char detector_name[128]    = "SURF";
char extractor_name[128]   = "SURF";
char matcher_name[128]     = "FlannBased";
char xml_name[128];

class Image_Recognition
{
	public:
		Ptr<DescriptorMatcher> matcher;
		Ptr<DescriptorExtractor> extractor;
		Ptr<FeatureDetector> detector;
		//HOGDescriptor hog;

		int vocab_sz;
		TermCriteria tc;
		int retries;
		int train_flags;

		BOWKMeansTrainer *bow_trainer;
		BOWImgDescriptorExtractor *bow_ide;
	
		map<int, pair<int,int> >mapp;
		vector< vector<int> > invertedIndex;

		Image_Recognition();
		void build_vocab(const path &);
		int descriptor_count(); 
		Mat cluster();
		void set_vocab(Mat &);
		void find_words_from_vocab(const path&, Mat &, vector< pair<int,int> > &);
		void make_invertedIndex(Mat &, vector< pair<int,int> > &);
		void find_words_test(const path &, Mat &);
		void find_docs(Mat &, set<int> &);
		void find_top8(Mat &, Mat &, set<int> &, vector<int> &);
};

Image_Recognition::Image_Recognition()
{
	detector    = FeatureDetector::create(detector_name);
	extractor   = DescriptorExtractor::create(extractor_name);
	matcher     = DescriptorMatcher::create(matcher_name);

	vocab_sz    = 1000;
	tc          = TermCriteria(CV_TERMCRIT_ITER, 10, 0.001);
	retries     = 1;
	train_flags = KMEANS_PP_CENTERS;

	bow_trainer = new BOWKMeansTrainer(vocab_sz, tc, retries, train_flags);
	bow_ide     = new BOWImgDescriptorExtractor(extractor, matcher);
}

void Image_Recognition::build_vocab(const path &basepath)
{
	for (recursive_directory_iterator dir(basepath); dir!=recursive_directory_iterator(); dir++)
	{
		path fname = dir->path();
		if (fname.extension() != ".jpg")
			continue;

		Mat img = imread(fname.string());
		if (img.empty())
		{
			cerr << "Warning: Could not read image: " << fname.string() << endl;
			continue;
		}

		vector<KeyPoint> keypoints;
		detector->detect(img, keypoints);
		if (keypoints.empty())
		{
			cerr << "Warning: Could not find key points in image: " << fname.string() << endl;
			continue;
		}
		Mat features;
		extractor->compute(img, keypoints, features); // all features have same dimensions
		/*
		   Mat img2;
		   cvtColor(img, img2, CV_RGB2GRAY);
		   vector<float> desc;
		   vector<Point> locations;
		   hog.compute(img2,desc,Size(24,24),Size(24,24),locations);
		   Mat features(desc);
		   cout << desc.size() << endl;
		   cout << features.rows << " " << features.cols << endl;
		 */
		bow_trainer->add(features);
	}
}

int Image_Recognition::descriptor_count()
{
	int cnt = 0;
	vector<Mat> descriptors = bow_trainer->getDescriptors();

	tr(descriptors, it)
		cnt += it->rows;
	return cnt;
}

Mat Image_Recognition::cluster()
{
	Mat ret = bow_trainer->cluster(); // feature dimensions remain same. Number of features = vocab size
	return ret;

}

void Image_Recognition::set_vocab(Mat &voc)
{
	bow_ide->setVocabulary(voc);
}

void Image_Recognition::find_words_from_vocab(const path& basepath, Mat &data, vector< pair<int,int> > &labels)
{
	for(recursive_directory_iterator dir(basepath); dir!=recursive_directory_iterator(); dir++) 
	{
		path fname = dir->path();
		if (fname.extension() != ".jpg") 
			continue;

		Mat img = imread(fname.string());
		if (img.empty()) 
		{
			cerr << "Warning: Could not read image: " << fname.string() << endl;
			continue;
		}
		
		vector<KeyPoint> keypoints;
		detector->detect(img, keypoints);
		if (keypoints.empty()) 
		{
			cerr << "Warning: Could not find key points in image: " << fname.string() << endl;
			continue;
		}
		int classno, imgno;
		Mat words;  // 1 row. length of row = vocab size
		bow_ide->compute(img, keypoints, words);
		data.push_back(words);   
		sscanf(fname.filename().c_str(),"%d_%d",&classno,&imgno);
		labels.push_back(make_pair(classno,imgno));
	} 
}

void Image_Recognition::make_invertedIndex(Mat &data, vector< pair<int,int> > &labels)
{
	int rows = data.rows;
	int cols = data.cols;
	invertedIndex = vector< vector<int> >(cols);
	
	for(int i=0;i<rows;i++)
	{
		mapp[i] = labels[i];
		for(int j=0;j<cols;j++)
			if(data.at<float>(i,j) != 0)
				invertedIndex[j].push_back(i);
	}
}

void Image_Recognition::find_docs(Mat &words, set<int> &doc_idx)
{
	for(int i=0; i<vocab_sz; i++)
		if(words.at<float>(1,i) != 0)
			tr(invertedIndex[i], it)
				doc_idx.insert(*it);
}

void Image_Recognition::find_top8(Mat &words, Mat &data, set<int> &doc_idx, vector<int> &top8)
{
	set< pair<float,int> > s;
	
	tr(doc_idx, it)
	{
		Mat dataw(data, Rect(0,*it,data.cols,1));
		float coss = words.dot(dataw) / ( norm(words)*norm(dataw) );
		
		if(s.size() < 8)
			s.insert(make_pair(coss,*it));
		else if(coss > (*s.begin()).first)
		{
			s.erase(s.begin());
			s.insert(make_pair(coss,*it));
		}
	}
	tr(s, it)
		top8.push_back(it->second);
}

void Image_Recognition::find_words_test(const path& basepath, Mat &data)
{
	for(recursive_directory_iterator dir(basepath); dir!=recursive_directory_iterator(); dir++) 
	{
		path fname = dir->path();
		if (fname.extension() != ".jpg") 
			continue;

		Mat img = imread(fname.string());
		if (img.empty()) 
		{
			cerr << "Warning: Could not read image: " << fname.string() << endl;
			continue;
		}
		
		vector<KeyPoint> keypoints;
		detector->detect(img, keypoints);
		if (keypoints.empty()) 
		{
			cerr << "Warning: Could not find key points in image: " << fname.string() << endl;
			continue;
		}
		int classno,imgno;
		Mat words;  // 1 row. length of row = vocab size
		set<int> doc_idx;
		vector<int> top8;

		
		sscanf(fname.filename().c_str(),"%d_%d",&classno,&imgno);
		bow_ide->compute(img, keypoints, words);
		find_docs(words, doc_idx);
		find_top8(words, data, doc_idx, top8);

		bool flag = false;
		int cnt = 1;
		for(int i=top8.size()-1; i>=0; i--)
		{
			if(mapp[top8[i]].first == classno)
				flag = true;
			if(display_flag)
			{
				char temp[128], temp2[128];
				sprintf(temp,"%s%03d_%04d.jpg",all_data, mapp[top8[i]].first, mapp[top8[i]].second);
				sprintf(temp2,"out/%d.png",cnt);
				++cnt;
				Mat tmp = imread(temp);
				Size s(350*tmp.cols/tmp.rows,350);
				resize(tmp,tmp,s);
				imwrite(temp2, tmp);
			}
		}
		if(flag)
			++succ_q;
		++num_q;
	} 
}

void save_data(int &vocab_sz, Mat &voc, Mat &train_data, vector< pair<int,int> > &labels)
{
		FileStorage fs(xml_name, FileStorage::WRITE);
		
		int l = labels.size();
		fs << "labels" << "[";
		for(int i = 0; i<l; i++)
			fs << "{:" << "x" << labels[i].first << "y" << labels[i].second << "}";
		fs << "]";

		fs << "vocab_sz" << vocab_sz;
		fs << "voc" << voc;
		fs << "train_data" << train_data;

		fs.release();
}

void load_data(int &vocab_sz, Mat &voc, Mat &train_data, vector< pair<int,int> > &labels)
{
		FileStorage fs(xml_name, FileStorage::READ);

		fs["vocab_sz"] >> vocab_sz;
		fs["voc"] >> voc;
		fs["train_data"] >> train_data;

		FileNode features = fs["labels"];
		for(FileNodeIterator it=features.begin(); it!=features.end(); it++)
			labels.push_back(make_pair( (int)(*it)["x"], (int)(*it)["y"] ));

		fs.release();
}

int main(int argc, char * argv[])
{
	if(argc < 3)
	{
		puts("Usage: ./a.out <0/1> <xml>");
		puts("Usage: ./a.out <0/1> <xml> <img>");
		return 0;
	}
	// Need to be called for non free Algorithm, else ::create won't work
	initModule_nonfree();
	trainFLAG = atoi(argv[1]);
	strcpy(xml_name, argv[2]);
	if(argc == 4)
	{
		strcpy(EVAL_DATA_PATH, argv[3]);
		display_flag = 1;
	}


	Image_Recognition ir;
	Mat voc;
	Mat train_data(0, ir.vocab_sz, CV_32FC1);
	vector< pair<int,int> > labels;
	
	cout << detector_name << endl;
	cout << extractor_name << endl;
	cout << matcher_name << endl;

	if(trainFLAG)
	{
		cout << "vocab sz = " << ir.vocab_sz << endl;
		cout << "Creating Vocabulary..." << endl;
		ir.build_vocab(path(TRAIN_DATA_PATH));

		cout << "Clustering " << ir.descriptor_count() << " features" << endl;
		voc = ir.cluster();
		cout << "Setting Vocab..." << endl;
		ir.set_vocab(voc);

		cout << "Processing training data..." << endl;
		ir.find_words_from_vocab(path(TRAIN_DATA_PATH), train_data, labels);

		cout << "Saving all data..." << endl;
		save_data(ir.vocab_sz, voc, train_data, labels);
	}
	else
	{
		cout << "Loading all data..." << endl;
		load_data(ir.vocab_sz, voc, train_data, labels);
		ir.set_vocab(voc);
	}
	cout << "Creating Inverted Index" << endl;
	ir.make_invertedIndex(train_data, labels);

	cout << "Testing" << endl;
	ir.find_words_test(path(EVAL_DATA_PATH), train_data);
	cout << "succ_q = " << succ_q << endl;
	cout << "num_q = " << num_q << endl;
	return 0;
}
