#include <iostream>
#include <algorithm>
#include <stdio.h>
#include <math.h>
#include <vector>
#include <numeric>
#include <strings.h>
#include <assert.h>
#include <png++/png.hpp>

using namespace std;

/*=======================================================================
STATIC EVALUATION PARAMETERS
=======================================================================*/

// holds the number of test images on the server
//const int32_t N_TESTIMAGES = 7518;
int32_t N_TESTIMAGES = 7480;

const int32_t append_zeros = 6;

// easy, moderate and hard evaluation level
enum DIFFICULTY{EASY=0, MODERATE=1, HARD=2};

// evaluation parameter
const int32_t MIN_HEIGHT[3]     = {40, 25, 25};     // minimum height for evaluated groundtruth/detections
const int32_t MAX_OCCLUSION[3]  = {0, 1, 2};        // maximum occlusion level of the groundtruth used for evaluation
const double  MAX_TRUNCATION[3] = {0.15, 0.3, 0.5}; // maximum truncation level of the groundtruth used for evaluation

// evaluated object classes
enum CLASSES{CAR=0, PEDESTRIAN=1, CYCLIST=2, VAN=3, TRUCK=4, PERSON_SITTING=5, TRAM=6};

// parameters varying per class
vector<string> CLASS_NAMES;
const double   MIN_OVERLAP[9] = {0.7, 0.5, 0.5, 0.7, 0.7, 0.5, 0.7};                  // the minimum overlap required for evaluation

// no. of recall steps that should be evaluated (discretized)
const double N_SAMPLE_PTS = 41;

// initialize class names
void initGlobals () {
  CLASS_NAMES.push_back("car");
  CLASS_NAMES.push_back("pedestrian");
  CLASS_NAMES.push_back("cyclist");
  CLASS_NAMES.push_back("van");
  CLASS_NAMES.push_back("truck");
  CLASS_NAMES.push_back("person_sitting");
  CLASS_NAMES.push_back("tram");
}

/*=======================================================================
DATA TYPES FOR EVALUATION
=======================================================================*/

// holding data needed for precision-recall and precision-aos
struct tPrData {
  vector<double> v;           // detection score for computing score thresholds
  double         similarity;  // orientation similarity
  int32_t        tp;          // true positives
  int32_t        fp;          // false positives
  int32_t        fn;          // false negatives
  tPrData () :
    similarity(0), tp(0), fp(0), fn(0) {}
};

// holding bounding boxes for ground truth and detections
struct tBox {
  string  type;     // object type as car, pedestrian or cyclist,...
  double   x1;      // left corner
  double   y1;      // top corner
  double   x2;      // right corner
  double   y2;      // bottom corner
  double   alpha;   // image orientation
  tBox (string type, double x1,double y1,double x2,double y2,double alpha) :
    type(type),x1(x1),y1(y1),x2(x2),y2(y2),alpha(alpha) {}
};

// holding ground truth data
struct tGroundtruth {
  tBox    box;        // object type, box, orientation
  double  truncation; // truncation 0..1
  int32_t occlusion;  // occlusion 0,1,2 (non, partly, fully)
  tGroundtruth () :
    box(tBox("invalild",-1,-1,-1,-1,-10)),truncation(-1),occlusion(-1) {}
  tGroundtruth (tBox box,double truncation,int32_t occlusion) :
    box(box),truncation(truncation),occlusion(occlusion) {}
  tGroundtruth (string type,double x1,double y1,double x2,double y2,double alpha,double truncation,int32_t occlusion) :
    box(tBox(type,x1,y1,x2,y2,alpha)),truncation(truncation),occlusion(occlusion) {}
};

// holding detection data
struct tDetection {
  tBox    box;    // object type, box, orientation
  double  thresh; // detection score
  tDetection ():
    box(tBox("invalid",-1,-1,-1,-1,-10)),thresh(-1000) {}
  tDetection (tBox box,double thresh) :
    box(box),thresh(thresh) {}
  tDetection (string type,double x1,double y1,double x2,double y2,double alpha,double thresh) :
    box(tBox(type,x1,y1,x2,y2,alpha)),thresh(thresh) {}
};

/*=======================================================================
FUNCTIONS TO LOAD DETECTION AND GROUND TRUTH DATA ONCE, SAVE RESULTS
=======================================================================*/

vector<tDetection> loadDetections(string file_name, bool &compute_aos, std::vector<bool> &eval_class, bool &success, vector<int> &count) {

  // holds all detections (ignored detections are indicated by an index vector
  vector<tDetection> detections;
  FILE *fp = fopen(file_name.c_str(),"r");
  if (!fp) {
    success = false;
    return detections;
  }
  while (!feof(fp)) {
    tDetection d;
    double trash;
    char str[255];
    if (fscanf(fp, "%s %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf",
                   str, &trash,    &trash,    &d.box.alpha,
                   &d.box.x1,   &d.box.y1, &d.box.x2, &d.box.y2,
                   &trash,      &trash,    &trash,    &trash,
                   &trash,      &trash,    &trash,    &d.thresh )==16) {
      d.box.type = str;
      detections.push_back(d);

      // orientation=-10 is invalid, AOS is not evaluated if at least one orientation is invalid
      if(d.box.alpha==-10)
        compute_aos = false;

      // a class is only evaluated if it is detected at least once
      for (int cls_ix=0; cls_ix<CLASS_NAMES.size(); cls_ix++){
        if (!strcasecmp(d.box.type.c_str(), CLASS_NAMES[cls_ix].c_str())){
          count[cls_ix]++;
          if(!eval_class[cls_ix])
            eval_class[cls_ix] = true;
          break;
        }
      }
    }
  }
  fclose(fp);
  success = true;
  return detections;
}

vector<tGroundtruth> loadGroundtruth(string file_name,bool &success, vector<int> &count) {

  // holds all ground truth (ignored ground truth is indicated by an index vector
  vector<tGroundtruth> groundtruth;
  FILE *fp = fopen(file_name.c_str(),"r");
  if (!fp) {
    success = false;
    return groundtruth;
  }
  while (!feof(fp)) {
    tGroundtruth g;
    double trash;
    char str[255];
    if (fscanf(fp, "%s %lf %d %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf",
                   str, &g.truncation, &g.occlusion, &g.box.alpha,
                   &g.box.x1,   &g.box.y1,     &g.box.x2,    &g.box.y2,
                   &trash,      &trash,        &trash,       &trash,
                   &trash,      &trash,        &trash )==15) {
      g.box.type = str;
      groundtruth.push_back(g);
    }
    for (int cls_idx=0; cls_idx<CLASS_NAMES.size(); cls_idx++){
      CLASSES cls = static_cast<CLASSES>(cls_idx);
      if(!strcasecmp(g.box.type.c_str(), CLASS_NAMES[cls].c_str())){
        count[cls_idx]++;
        break;
      }
    }
  }
  fclose(fp);
  success = true;
  return groundtruth;
}

void saveStats (const vector<double> &precision, const vector<double> &aos, FILE *fp_det, FILE *fp_ori) {

  // save precision to file
  if(precision.empty())
    return;
  for (int32_t i=0; i<precision.size(); i++)
    fprintf(fp_det,"%f ",precision[i]);
  fprintf(fp_det,"\n");

  // save orientation similarity, only if there were no invalid orientation entries in submission (alpha=-10)
  if(aos.empty())
    return;
  for (int32_t i=0; i<aos.size(); i++)
    fprintf(fp_ori,"%f ",aos[i]);
  fprintf(fp_ori,"\n");
}

/*=======================================================================
EVALUATION HELPER FUNCTIONS
=======================================================================*/

// criterion defines whether the overlap is computed with respect to both areas (ground truth and detection)
// or with respect to box a or b (detection and "dontcare" areas)
inline double boxoverlap(tBox a, tBox b, int32_t criterion=-1){

  // overlap is invalid in the beginning
  double o = -1;

  // get overlapping area
  double x1 = max(a.x1, b.x1);
  double y1 = max(a.y1, b.y1);
  double x2 = min(a.x2, b.x2);
  double y2 = min(a.y2, b.y2);

  // compute width and height of overlapping area
  double w = x2-x1;
  double h = y2-y1;

  // set invalid entries to 0 overlap
  if(w<=0 || h<=0)
    return 0;

  // get overlapping areas
  double inter = w*h;
  double a_area = (a.x2-a.x1) * (a.y2-a.y1);
  double b_area = (b.x2-b.x1) * (b.y2-b.y1);

  // intersection over union overlap depending on users choice
  if(criterion==-1)     // union
    o = inter / (a_area+b_area-inter);
  else if(criterion==0) // bbox_a
    o = inter / a_area;
  else if(criterion==1) // bbox_b
    o = inter / b_area;

  // overlap
  return o;
}

vector<double> getThresholds(vector<double> &v, double n_groundtruth){

  // holds scores needed to compute N_SAMPLE_PTS recall values
  vector<double> t;

  // sort scores in descending order
  // (highest score is assumed to give best/most confident detections)
  sort(v.begin(), v.end(), greater<double>());

  // get scores for linearly spaced recall
  double current_recall = 0;
  for(int32_t i=0; i<v.size(); i++){

    // check if right-hand-side recall with respect to current recall is close than left-hand-side one
    // in this case, skip the current detection score
    double l_recall, r_recall, recall;
    l_recall = (double)(i+1)/n_groundtruth;
    if(i<(v.size()-1))
      r_recall = (double)(i+2)/n_groundtruth;
    else
      r_recall = l_recall;

    if( (r_recall-current_recall) < (current_recall-l_recall) && i<(v.size()-1))
      continue;

    // left recall is the best approximation, so use this and goto next recall step for approximation
    recall = l_recall;

    // the next recall step was reached
    t.push_back(v[i]);
    current_recall += 1.0/(N_SAMPLE_PTS-1.0);
  }
  return t;
}

void cleanData(CLASSES current_class, const vector<tGroundtruth> &gt, const vector<tDetection> &det, vector<int32_t> &ignored_gt, vector<tGroundtruth> &dc, vector<int32_t> &ignored_det, int32_t &n_gt, DIFFICULTY difficulty){

  // extract ground truth bounding boxes for current evaluation class
  for(int32_t i=0;i<gt.size(); i++){

    // only bounding boxes with a minimum height are used for evaluation
    double height = gt[i].box.y2 - gt[i].box.y1;

    // neighboring classes are ignored ("van" for "car" and "person_sitting" for "pedestrian")
    // (lower/upper cases are ignored)
    int32_t valid_class;

    // all classes without a neighboring class
    if(!strcasecmp(gt[i].box.type.c_str(), CLASS_NAMES[current_class].c_str()))
      valid_class = 1;

    // classes with a neighboring class
    else if(!strcasecmp(CLASS_NAMES[current_class].c_str(), "Pedestrian") && !strcasecmp("Person_sitting", gt[i].box.type.c_str()))
      valid_class = 0;
    else if(!strcasecmp(CLASS_NAMES[current_class].c_str(), "Car") && !strcasecmp("Van", gt[i].box.type.c_str()))
      valid_class = 0;
    else if (!strcasecmp(CLASS_NAMES[current_class].c_str(), "Person_sitting") && !strcasecmp("Pedestrian", gt[i].box.type.c_str()))
      valid_class = 0;
    else if (!strcasecmp(CLASS_NAMES[current_class].c_str(), "Van") && !strcasecmp("Car", gt[i].box.type.c_str()))
      valid_class = 0;
    // classes not used for evaluation
    else
      valid_class = -1;

    // ground truth is ignored, if occlusion, truncation exceeds the difficulty or ground truth is too small
    // (doesn't count as FN nor TP, although detections may be assigned)
    bool ignore = false;
    if(gt[i].occlusion>MAX_OCCLUSION[difficulty] || gt[i].truncation>MAX_TRUNCATION[difficulty] || height<MIN_HEIGHT[difficulty])
      ignore = true;

    // set ignored vector for ground truth
    // current class and not ignored (total no. of ground truth is detected for recall denominator)
    if(valid_class==1 && !ignore){
      ignored_gt.push_back(0);
      n_gt++;
    }

    // neighboring class, or current class but ignored
    else if(valid_class==0 || (ignore && valid_class==1))
      ignored_gt.push_back(1);

    // all other classes which are FN in the evaluation
    else
      ignored_gt.push_back(-1);
  }

  // extract dontcare areas
  for(int32_t i=0;i<gt.size(); i++)
    if(!strcasecmp("DontCare", gt[i].box.type.c_str()))
      dc.push_back(gt[i]);

  // extract detections bounding boxes of the current class
  for(int32_t i=0;i<det.size(); i++){

    // neighboring classes are not evaluated
    int32_t valid_class;
    if(!strcasecmp(det[i].box.type.c_str(), CLASS_NAMES[current_class].c_str()))
      valid_class = 1;
    else
      valid_class = -1;

    int32_t height = fabs(det[i].box.y1 - det[i].box.y2);

    // set ignored vector for detections
    if(height<MIN_HEIGHT[difficulty])
      ignored_det.push_back(1);
    else if(valid_class==1)
      ignored_det.push_back(0);
    else
      ignored_det.push_back(-1);
  }
}

void addRectangle(png::image<png::gray_pixel> &M, tBox box) {
  int32_t width = M.get_width();
  int32_t height = M.get_height();
  for (int32_t v=max((int)box.y1,0); v<=min((int)box.y2,height-1); v++)
    for (int32_t u=max((int)box.x1,0); u<=min((int)box.x2,width-1); u++)
      //M.set_pixel(u,v,min((int)M.get_pixel(u,v)+1,255)); // problematic at overlapping areas
      M.set_pixel(u,v,1);
}

png::image<png::gray_pixel> addMap (png::image<png::gray_pixel> M1, png::image<png::gray_pixel> M2) {
  int32_t width = M1.get_width();
  int32_t height = M1.get_height();
  png::image<png::gray_pixel> M(width,height);
  for (int32_t v=0; v<height-0; v++)
    for (int32_t u=0; u<width-0; u++)
      M.set_pixel(u,v,min((int)M1.get_pixel(u,v)+(int)M2.get_pixel(u,v),255));
  return M;
}

tPrData computeStatistics(int frame, CLASSES current_class, const vector<tGroundtruth> &gt, const vector<tDetection> &det, const vector<tGroundtruth> &dc, const vector<int32_t> &ignored_gt, const vector<int32_t>  &ignored_det, bool compute_fp, bool compute_aos=false, double thresh=0, bool debug=false, int top_stats=0){

  tPrData stat = tPrData();
  const double NO_DETECTION = -10000000;
  vector<double> delta;            // holds angular difference for TPs (needed for AOS evaluation)
  vector<bool> assigned_detection; // holds wether a detection was assigned to a valid or ignored ground truth
  assigned_detection.assign(det.size(), false);
  vector<bool> ignored_threshold;
  ignored_threshold.assign(det.size(), false); // holds detections with a threshold lower than thresh if FP are computed

  // init top stats images
  int width = 1242;
  int height = 376;
  png::image<png::gray_pixel> M_tp, M_fp, M_fn;
  if (top_stats>0) {
    M_tp = png::image<png::gray_pixel>(width,height);
    M_fp = png::image<png::gray_pixel>(width,height);
    M_fn = png::image<png::gray_pixel>(width,height);
  }

  // detections with a low score are ignored for computing precision (needs FP)
  if(compute_fp)
    for(int32_t i=0; i<det.size(); i++)
      if(det[i].thresh<thresh)
        ignored_threshold[i] = true;

  // evaluate all ground truth boxes
  for(int32_t i=0; i<gt.size(); i++){

    // this ground truth is not of the current or a neighboring class and therefore ignored
    if(ignored_gt[i]==-1)
      continue;

    /*=======================================================================
    find candidates (overlap with ground truth > 0.5) (logical len(det))
    =======================================================================*/
    int32_t det_idx          = -1;
    double valid_detection = NO_DETECTION;
    double max_overlap     = 0;

    // search for a possible detection
    bool assigned_ignored_det = false;
    for(int32_t j=0; j<det.size(); j++){

      // detections not of the current class, already assigned or with a low threshold are ignored
      if(ignored_det[j]==-1)
        continue;
      if(assigned_detection[j])
        continue;
      if(ignored_threshold[j])
        continue;

      // find the maximum score for the candidates and get idx of respective detection
      double overlap = boxoverlap(det[j].box, gt[i].box);

      // for computing recall thresholds, the candidate with highest score is considered
      if(!compute_fp && overlap>MIN_OVERLAP[current_class] && det[j].thresh>valid_detection){
        det_idx         = j;
        valid_detection = det[j].thresh;
      }

      // for computing pr curve values, the candidate with the greatest overlap is considered
      // if the greatest overlap is an ignored detection (min_height), the overlapping detection is used
      else if(compute_fp && overlap>MIN_OVERLAP[current_class] && (overlap>max_overlap || assigned_ignored_det) && ignored_det[j]==0){
        max_overlap     = overlap;
        det_idx         = j;
        valid_detection = 1;
        assigned_ignored_det = false;
      }
      else if(compute_fp && overlap>MIN_OVERLAP[current_class] && valid_detection==NO_DETECTION && ignored_det[j]==1){
        det_idx              = j;
        valid_detection      = 1;
        assigned_ignored_det = true;
      }
    }

    /*=======================================================================
    compute TP, FP and FN
    =======================================================================*/

    // nothing was assigned to this valid ground truth
    if(valid_detection==NO_DETECTION && ignored_gt[i]==0) {
      stat.fn++;

      // draw rectangle of ones onto fn image at gt[i].box
      if (top_stats>0)
        addRectangle(M_fn,gt[i].box);
    }

    // only evaluate valid ground truth <=> detection assignments (considering difficulty level)
    else if(valid_detection!=NO_DETECTION && (ignored_gt[i]==1 || ignored_det[det_idx]==1))
      assigned_detection[det_idx] = true;

    // found a valid true positive
    else if(valid_detection!=NO_DETECTION){

      // write highest score to threshold vector
      stat.tp++;
      stat.v.push_back(det[det_idx].thresh);

      // draw rectangle of ones onto tp image at gt[i].box
      if (top_stats>0)
        addRectangle(M_tp,gt[i].box);

      // compute angular difference of detection and ground truth if valid detection orientation was provided
      if(compute_aos)
        delta.push_back(gt[i].box.alpha - det[det_idx].box.alpha);

      // clean up
      assigned_detection[det_idx] = true;
    }
  }

  // compute false positives (considering stuff area)
  if(compute_fp){

    // count fp
    for(int32_t i=0; i<det.size(); i++){

      // if not yet assigned and not ignored
      if(!(assigned_detection[i] || ignored_det[i]==-1 || ignored_det[i]==1 || ignored_threshold[i])) {

        // check if it overlaps with dontcare area
        for(int32_t j=0; j<dc.size(); j++){

          // does detection overlap with dontcare area?
          double overlap = boxoverlap(det[i].box, dc[j].box, 0);
          if (overlap>MIN_OVERLAP[current_class])
            assigned_detection[i] = true;
        }

        // if not assigned to gt box or dontcare area we have a false positive detection
        if (!assigned_detection[i]) {
          stat.fp++;

          // draw rectangle of ones onto fp image at det[i].box
          if (top_stats>0)
            addRectangle(M_fp,det[i].box);
        }
      }
    }

    // if all orientation values are valid, the AOS is computed
    if(compute_aos){
      vector<double> tmp;

      // FP have a similarity of 0, for all TP compute AOS
      tmp.assign(stat.fp, 0);
      for(int32_t i=0; i<delta.size(); i++)
        tmp.push_back((1.0+cos(delta[i]))/2.0);

      // be sure, that all orientation deltas are computed
      assert(tmp.size()==stat.fp+stat.tp);
      assert(delta.size()==stat.tp);

      // get the mean orientation similarity for this image
      if(stat.tp>0 || stat.fp>0)
        stat.similarity = accumulate(tmp.begin(), tmp.end(), 0.0);

      // there was neither a FP nor a TP, so the similarity is ignored in the evaluation
      else
        stat.similarity = -1;
    }
  }

  // save top stats images (M_tp,M_fp,M_fn)
  if (top_stats>0) {

    // assemble file names
    char prefix[256];
    sprintf(prefix,"%06d",frame);
    string top_dir = ""; // warning: naming appears twice in this file
    if      (top_stats==1) top_dir = "results_top/object_car";
    else if (top_stats==2) top_dir = "results_top/object_pedestrian";
    else if (top_stats==3) top_dir = "results_top/object_cyclist";
    string M_tp_file_name = top_dir + "/tp/" + prefix + ".png";
    string M_fp_file_name = top_dir + "/fp/" + prefix + ".png";
    string M_fn_file_name = top_dir + "/fn/" + prefix + ".png";

    // load and add maps
    png::image<png::gray_pixel> M_tp_file, M_fp_file, M_fn_file;
    bool read_error = false;
    try {
      M_tp_file = png::image<png::gray_pixel>(M_tp_file_name);
      M_fp_file = png::image<png::gray_pixel>(M_fp_file_name);
      M_fn_file = png::image<png::gray_pixel>(M_fn_file_name);
    } catch (...) {read_error = true;}
    if (!read_error) {
      M_tp = addMap(M_tp,M_tp_file);
      M_fp = addMap(M_fp,M_fp_file);
      M_fn = addMap(M_fn,M_fn_file);
    }

    // write maps
    M_tp.write(M_tp_file_name);
    M_fp.write(M_fp_file_name);
    M_fn.write(M_fn_file_name);
  }

  return stat;
}

/*=======================================================================
EVALUATE CLASS-WISE
=======================================================================*/

bool eval_class (FILE *fp_det, FILE *fp_ori, CLASSES current_class,const vector< vector<tGroundtruth> > &groundtruth,const vector< vector<tDetection> > &detections, bool compute_aos, vector<double> &precision, vector<double> &aos, DIFFICULTY difficulty, int top_stats) {

  // top statistics is only evaluated for MODERATE mode
  if (top_stats>0 && difficulty!=MODERATE)
    return true;

  // init
  int32_t n_gt=0;                                     // total no. of gt (denominator of recall)
  vector<double> v, thresholds;                       // detection scores, evaluated for recall discretization
  vector< vector<int32_t> > ignored_gt, ignored_det;  // index of ignored gt detection for current class/difficulty
  vector< vector<tGroundtruth> > dontcare;            // index of dontcare areas, included in ground truth

  std::cout << "Getting detection scores to compute thresholds" << std::endl;
  // for all test images do (get detection scores to compute thresholds)
  for (int32_t i=0; i<N_TESTIMAGES; i++){

    // holds ignored ground truth, ignored detections and dontcare areas for current frame
    vector<int32_t> i_gt, i_det;
    vector<tGroundtruth> dc;

    // only evaluate objects of current class and ignore occluded, truncated objects
    cleanData(current_class, groundtruth[i], detections[i], i_gt, dc, i_det, n_gt, difficulty);
    ignored_gt.push_back(i_gt);
    ignored_det.push_back(i_det);
    dontcare.push_back(dc);

    // compute statistics to get recall values
    tPrData pr_tmp = tPrData();
    pr_tmp = computeStatistics(i,current_class, groundtruth[i], detections[i], dc, i_gt, i_det, false);

    // add detection scores to vector over all images
    for(int32_t j=0; j<pr_tmp.v.size(); j++)
      v.push_back(pr_tmp.v[j]);
  }

  // get thresholds that must be evaluated for recall discretization
  thresholds = getThresholds(v, n_gt);

  // compute TP,FP,FN at all thresholds
  vector<tPrData> pr;
  pr.assign(thresholds.size(),tPrData());

  std::cout << "Computing statistics" << std::endl;
  for (int32_t i=0; i<N_TESTIMAGES; i++){

    // for all scores/recall thresholds do:
    for(int32_t t=0; t<thresholds.size(); t++){
      tPrData tmp = tPrData();
      tmp = computeStatistics(i,current_class, groundtruth[i], detections[i], dontcare[i],
                              ignored_gt[i], ignored_det[i], true, compute_aos, thresholds[t], t==38, 0);

      // add no. of TP, FP, FN, AOS for current frame to total evaluation for current threshold
      pr[t].tp += tmp.tp;
      pr[t].fp += tmp.fp;
      pr[t].fn += tmp.fn;
      if(tmp.similarity!=-1)
        pr[t].similarity += tmp.similarity;
    }
  }

  // compute recall, precision and AOS (average orientation similarity)
  vector<double> recall;
  precision.assign(N_SAMPLE_PTS, 0);
  if(compute_aos)
    aos.assign(N_SAMPLE_PTS, 0);
  double r=0;
  for (int32_t i=0; i<thresholds.size(); i++){
    r = pr[i].tp/(double)(pr[i].tp + pr[i].fn);
    recall.push_back(r);
    precision[i] = pr[i].tp/(double)(pr[i].tp + pr[i].fp);
    if(compute_aos)
      aos[i] = pr[i].similarity/(double)(pr[i].tp + pr[i].fp);
  }

  // filter precision and AOS using max_{i..end}(precision)
  for (int32_t i=0; i<thresholds.size(); i++){
    precision[i] = *max_element(precision.begin()+i, precision.end());
    if(compute_aos)
      aos[i] = *max_element(aos.begin()+i, aos.end());
  }

  // top statistics
  if (top_stats>0 && difficulty==MODERATE && current_class+1==top_stats) {

    // get score threshold at which precision = recall (approximately)
    int32_t t_opt = 0; float t_opt_dist = 1;
    for (int32_t i=0; i<thresholds.size(); i++){
      float t_dist = fabs(precision[i]-recall[i]);
      if (t_dist < t_opt_dist) {
        t_opt = i;
        t_opt_dist = t_dist;
      }
    }

    // create top stats directory
    string top_dir = "";
    if      (top_stats==1) top_dir = "results_top/object_car";
    else if (top_stats==2) top_dir = "results_top/object_pedestrian";
    else if (top_stats==3) top_dir = "results_top/object_cyclist";
    else top_stats=0;
    if (top_stats>0) {
      system(("mkdir -p " + top_dir).c_str());
      system(("mkdir -p " + top_dir + "/tp").c_str());
      system(("mkdir -p " + top_dir + "/fp").c_str());
      system(("mkdir -p " + top_dir + "/fn").c_str());
    }

    // init top stats table
    vector<tPrData> table;

    // compute TP,FP,FN for each image at threshold t_opt (precision=recall)
    std::cout << "Computing confusion matrix" << std::endl;
    for (int32_t i=0; i<N_TESTIMAGES; i++){
      tPrData pr = computeStatistics(i,current_class, groundtruth[i], detections[i], dontcare[i],
                                     ignored_gt[i], ignored_det[i], true, compute_aos, thresholds[t_opt],
                                     false, top_stats);
      table.push_back(pr);
    }

    // file name
    string top_stats_file_name = top_dir+"/stats.txt";

    // read from top stats file
    FILE *fp; tPrData val;
    fp = fopen(top_stats_file_name.c_str(),"r");
    if (fp) {
      for (int i=0; i<table.size(); i++) {
        if (fscanf(fp,"%d",&val.tp)<1) break;
        if (fscanf(fp,"%d",&val.fp)<1) break;
        if (fscanf(fp,"%d",&val.fn)<1) break;
        table[i].tp += val.tp;
        table[i].fp += val.fp;
        table[i].fn += val.fn;
      }
      fclose(fp);
    }

    // write to top stats file
    fp = fopen(top_stats_file_name.c_str(),"w");
    for (int i=0; i<table.size(); i++)
      fprintf(fp,"%d %d %d\n",table[i].tp,table[i].fp,table[i].fn);
    fclose(fp);
  }

  // save statisics and finish with success
  if (top_stats==0)
    saveStats(precision, aos, fp_det, fp_ori);
	return true;
}

void saveAndPlotPlots(string dir_name,string file_name,string obj_type,vector<double> vals[],bool is_aos){

  char command[1024];

  // save plot data to file
  FILE *fp = fopen((dir_name + "/" + file_name + ".txt").c_str(),"w");
  for (int32_t i=0; i<(int)N_SAMPLE_PTS; i++)
    fprintf(fp,"%f %f %f %f\n",(double)i/(N_SAMPLE_PTS-1.0),vals[0][i],vals[1][i],vals[2][i]);
  fclose(fp);

  // create png + eps
  for (int32_t j=0; j<2; j++) {

    // open file
    FILE *fp = fopen((dir_name + "/" + file_name + ".gp").c_str(),"w");

    // save gnuplot instructions
    if (j==0) {
      fprintf(fp,"set term png size 450,315 font \"Helvetica\" 11\n");
      fprintf(fp,"set output \"%s.png\"\n",file_name.c_str());
    } else {
      fprintf(fp,"set term postscript eps enhanced color font \"Helvetica\" 20\n");
      fprintf(fp,"set output \"%s.eps\"\n",file_name.c_str());
    }

    // set labels and ranges
    fprintf(fp,"set size ratio 0.7\n");
    fprintf(fp,"set xrange [0:1]\n");
    fprintf(fp,"set yrange [0:1]\n");
    fprintf(fp,"set xlabel \"Recall\"\n");
    if (!is_aos) fprintf(fp,"set ylabel \"Precision\"\n");
    else         fprintf(fp,"set ylabel \"Orientation Similarity\"\n");
    obj_type[0] = toupper(obj_type[0]);
    fprintf(fp,"set title \"%s\"\n",obj_type.c_str());

    // line width
    int32_t   lw = 5;
    if (j==0) lw = 3;

    // plot error curve
    fprintf(fp,"plot ");
    fprintf(fp,"\"%s.txt\" using 1:2 title 'Easy' with lines ls 1 lw %d,",file_name.c_str(),lw);
    fprintf(fp,"\"%s.txt\" using 1:3 title 'Moderate' with lines ls 2 lw %d,",file_name.c_str(),lw);
    fprintf(fp,"\"%s.txt\" using 1:4 title 'Hard' with lines ls 3 lw %d",file_name.c_str(),lw);

    // close file
    fclose(fp);

    // run gnuplot => create png + eps
    sprintf(command,"cd %s; gnuplot %s",dir_name.c_str(),(file_name + ".gp").c_str());
    system(command);
  }

  // create pdf and crop
  sprintf(command,"cd %s; ps2pdf %s.eps %s_large.pdf",dir_name.c_str(),file_name.c_str(),file_name.c_str());
  system(command);
  sprintf(command,"cd %s; pdfcrop %s_large.pdf %s.pdf",dir_name.c_str(),file_name.c_str(),file_name.c_str());
  system(command);
  sprintf(command,"cd %s; rm %s_large.pdf",dir_name.c_str(),file_name.c_str());
  system(command);
}

bool eval(string result_sha,string input_dataset,int top_stats){

  // set some global parameters
  initGlobals();

  // ground truth and result directories
  string gt_dir         = "data/object/label_2";
  string result_dir     = "results/" + result_sha;
  string plot_dir       = result_dir + "/plot";
  string valid_imgs_path = "lists/" + input_dataset + ".txt";

  std::cout << "Results list: " << valid_imgs_path << std::endl;

  // create output directories
  system(("mkdir " + plot_dir).c_str());

  // hold detections and ground truth in memory
  vector< vector<tGroundtruth> > groundtruth;
  vector< vector<tDetection> >   detections;

  // holds wether orientation similarity shall be computed (might be set to false while loading detections)
  // and which labels where provided by this submission
  bool compute_aos=true, eval_car=false, eval_pedestrian=false, eval_cyclist=false;

  std::cout << "Getting valid imagess... " << std::endl;
  // Get image indices
  ifstream valid_imgs(valid_imgs_path.c_str());
  if (!valid_imgs.is_open()){
    std::cout << valid_imgs_path << " not found" << std::endl;
    exit(-1);
  }
  string line;
  vector<int> indices;
  while (!valid_imgs.eof())
  {
    getline (valid_imgs,line);
    if (atoi(line.c_str())!=0){
      indices.push_back(atoi(line.c_str()));
    }
  }
  std::cout << "File loaded" << std::endl;

  N_TESTIMAGES = indices.size();

  // Just to get stats for each class
  vector<int> count, count_gt;
  count.assign(CLASS_NAMES.size(), 0);
  count_gt.assign(CLASS_NAMES.size(), 0);

  // for all images read groundtruth and detections
  std::cout << "Loading detections... " << std::endl;
  std::vector<bool> do_eval_class(CLASS_NAMES.size());

  for (int32_t i=0; i<N_TESTIMAGES; i++) {

    // file name
    char file_name[256];
    switch (append_zeros){
      case 6:
        sprintf(file_name,"%06d.txt",indices[i]);
        break;
      case 8:
        sprintf(file_name,"%08d.txt",indices[i]);
        break;
      default:
        std::cout << "ERROR: Undefined number of zeros to append" << std::endl;
    }

    // read ground truth and result poses
    bool gt_success,det_success;
    vector<tGroundtruth> gt   = loadGroundtruth(gt_dir + "/" + file_name, gt_success, count_gt);
    vector<tDetection>   det  = loadDetections(result_dir + "/data/" + file_name, compute_aos, do_eval_class, det_success,count);
    groundtruth.push_back(gt);
    detections.push_back(det);

    // check for errors
    if (!gt_success) {
      std::cout << "ERROR: Couldn't read: " << gt_dir + "/" + file_name << " of ground truth." << std::endl;
      return false;
    }
    if (!det_success) {
      std::cout << "ERROR: Couldn't read: " << result_dir + "/data/" + file_name <<std::endl;
      return false;
    }
  }
  // Print stats
  cout << "-----------" << endl;
  cout << "GT STATS" << endl;
  cout << "-----------" << endl;
  for (int cls_idx=0; cls_idx<CLASS_NAMES.size(); cls_idx++){
    cout << CLASS_NAMES[cls_idx].c_str() << " : " << count_gt[cls_idx] << endl;
  }
  cout << "-----------" << endl;
  cout << "DET STATS" << endl;
  cout << "-----------" << endl;
  for (int cls_idx=0; cls_idx<CLASS_NAMES.size(); cls_idx++){
    cout << CLASS_NAMES[cls_idx].c_str() << " : " << count[cls_idx] << endl;
  }
  std::cout << "  done." << std::endl;

  // holds pointers for result files
  FILE *fp_det=0, *fp_ori=0;

  for (int cls_ix=0; cls_ix<CLASS_NAMES.size(); cls_ix++){
    if (do_eval_class[cls_ix]){
      if (top_stats==0) {
        fp_det = fopen((result_dir + "/stats_" + CLASS_NAMES[cls_ix] + "_detection.txt").c_str(),"w");
        if(compute_aos)
          fp_ori = fopen((result_dir + "/stats_" + CLASS_NAMES[cls_ix] + "_orientation.txt").c_str(),"w");
      }

      vector<double> precision[3], aos[3];
      if (top_stats==0 || top_stats==1) {
        std::cout << "Evaluating " << CLASS_NAMES[cls_ix] << " ..." << std::endl;
        std::cout << "EASY" << std::endl;
        bool car_easy = eval_class(fp_det,fp_ori,static_cast<CLASSES>(cls_ix),groundtruth,detections,compute_aos,precision[0],aos[0],EASY,top_stats);
        std::cout << "MODERATE" << std::endl;
        bool car_mod = eval_class(fp_det,fp_ori,static_cast<CLASSES>(cls_ix),groundtruth,detections,compute_aos,precision[1],aos[1],MODERATE,top_stats);
        std::cout << "HARD" << std::endl;
        bool car_hard = eval_class(fp_det,fp_ori,static_cast<CLASSES>(cls_ix),groundtruth,detections,compute_aos,precision[2],aos[2],HARD,top_stats);
        if(!car_easy || !car_mod || !car_hard){
          std::cout << CLASS_NAMES[cls_ix] << " evaluation failed." << std::endl;
          return false;
        }
      }

      if (top_stats==0) {
        fclose(fp_det);
        saveAndPlotPlots(plot_dir,CLASS_NAMES[cls_ix] + "_detection",CLASS_NAMES[cls_ix],precision,0);
        if(compute_aos){
          saveAndPlotPlots(plot_dir,CLASS_NAMES[cls_ix] + "_orientation",CLASS_NAMES[cls_ix],aos,1);
          fclose(fp_ori);
        }
      }
    }
  }

  // success
  return true;
}

int32_t main (int32_t argc,char *argv[]) {

  // we need 2 or 4 arguments!
  if (argc!=2 && argc!=3 && argc!=4) {
    cout << "Usage: ./eval_detection result_sha val_dataset [top_stats]" << endl;
    return 1;
  }

  // read arguments
  string result_sha = argv[1];
  string input_dataset = argv[2];

  std:cout << "Starting evaluation..." << std::endl;

  // run evaluation
  int top_stats = 0;
  if (argc==4) top_stats = atoi(argv[3]);
  bool success = eval(result_sha,input_dataset,top_stats);

  return 0;
}
