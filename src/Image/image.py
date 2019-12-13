bgLearn  learn the backgroud by pics from cam then get a background model

vector<Vec4d> convertRoiCoordinate(vector<Vec4d> Lines,Rect Roi)

void SamplesEnhance::findrColor(Mat &imgO,Rect &rectout)


cvtColor(frame, gray, CV_RGB2GRAY);



Canny(theobj, theobj, 10, 50);

HoughLinesP(theobj, Lines, 1, CV_PI / 360, 60, 30, 5);

checkImage   check the cnn detected result by image process and image track and then update the bottle dict