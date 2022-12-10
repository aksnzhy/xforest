#include <iostream>
#include <string>
#include <fstream>
#include <sstream>
#include <chrono>
#include <vector>
#include <queue>
#include <cmath>
#include <numeric>

#ifndef NUM_DATA
#define NUM_DATA 1000000
#endif

#ifndef NUM_TEST_DATA
#define NUM_TEST_DATA 50000
#endif

#ifndef NUM_FEAT
#define NUM_FEAT 28
#endif

#ifndef MAX_DEPTH
#define MAX_DEPTH 12
#endif

#ifndef MIN_SPLIT
#define MIN_SPLIT 256
#endif

typedef float real_t;

const std::string tr_file("./higgs-train-1m.csv");
const std::string te_file("./higgs-test.csv");

void read_data(const std::string& filename, real_t* X, real_t* Y, size_t len) {
  std::ifstream train_fs;
  train_fs.open(filename.c_str());
  for (auto i = 0; i < len; ++i) {
    std::string line;
    std::getline(train_fs, line);
    std::istringstream ss(line);
    std::string number;
    std::getline(ss, number, ',');
    Y[i] = stof(number);
    for (auto j = 0; j < NUM_FEAT; ++j) {
      std::getline(ss, number, ',');
      X[i*NUM_FEAT+j] = stof(number);
    }
  }
  train_fs.close();
}

class Timer {
 public:
  Timer() {
    reset();
  }

  // Reset start time
  void reset() {
    begin = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::microseconds>(begin-begin);
  }

  // Code start
  void tic() {
    begin = std::chrono::high_resolution_clock::now();
  }

  // Code end
  float toc() {
    duration += std::chrono::duration_cast<std::chrono::microseconds>
              (std::chrono::high_resolution_clock::now()-begin);
    return get();
  }

  // Get the time duration
  float get() {
    return (float)duration.count() / 1000.0 / 1000.0;
  }

 protected:
    std::chrono::high_resolution_clock::time_point begin;
    std::chrono::microseconds duration;
};

//-----------------------------------------------------------------------------
// Random forest for classification
//-----------------------------------------------------------------------------

uint8_t max_bin = 255;
uint8_t num_class = 2;

float time_find = 0.0;
float time_split = 0.0;

//-----------------------------------------------------------------------------
// Find max and min value in data
//-----------------------------------------------------------------------------
struct MaxMin {
  real_t gap = 0.0;
  real_t max_feat = std::numeric_limits<real_t>::min();
  real_t min_feat = std::numeric_limits<real_t>::max();
};

//-----------------------------------------------------------------------------
// Histogram data structure
//-----------------------------------------------------------------------------
struct Histogram {
  // ctor and dctor
  Histogram(const uint32_t num_feat, 
            const uint32_t num_bin,
            const uint8_t num_class) {
    count_len = num_feat * num_bin * num_class;
    count = new uint32_t[count_len];
    for (uint32_t i = 0; i < count_len; ++i) {
      count[i] = 0;
    }
  }
  ~Histogram() {
    delete [] count;
  }
  uint32_t count_len = 0;
  uint32_t* count = nullptr;
};

//-----------------------------------------------------------------------------
// Tmp info during training
//-----------------------------------------------------------------------------
struct DTNode;

struct TInfo {
  // ctor and dctor
  TInfo(const uint32_t num_feat, 
        const uint32_t num_bin, 
        const uint8_t num_class) {
    histo = new Histogram(num_feat, num_bin, num_class);
  }
  ~TInfo() {
    delete histo;
  }
  // left or right
  char l_or_r;
  // node layer
  uint8_t level = 1;
  // start postion
  uint32_t start_pos = 0;
  // end position
  uint32_t end_pos = 0;
  // split position
  uint32_t mid_pos = 0;
  // Best gini value
  real_t best_gini = 1.0;
  // Parent node
  DTNode* parent = nullptr;
  // Brother node
  DTNode* brother = nullptr;
  // Histogram
  Histogram* histo = nullptr;
};

//-----------------------------------------------------------------------------
// Decision tree node
//-----------------------------------------------------------------------------
struct DTNode {
  bool is_leaf = false;
  // leaf node value
  real_t leaf_val = -1.0;
  // left child
  DTNode* l_child = nullptr;
  // right child
  DTNode* r_child = nullptr;
  // Best feature
  uint32_t best_feat_id = 0;
  // Best bin value
  uint8_t best_bin_val = 0;
  // Tmp info used by training
  TInfo *info = nullptr;
  // Initialize TInfo
  inline void Init(const uint32_t num_feat, 
                   const uint32_t num_bin, 
                   const uint8_t num_class) {
    info = new TInfo(num_feat, num_bin, num_class);
  }
  // Clear TInfo
  inline void Clear() {
    delete info;
  }
  // Clear Parent TInfo
  inline void ClearParent() {
    delete info->parent->info;
  }
  // Is a leaf node?
  inline bool IsLeaf() {
    return is_leaf;
  }
  inline void SetLeaf() {
    is_leaf = true;
  }
  // Leaf value
  inline real_t LeafVal() {
    return leaf_val;
  }
  inline void SetLeafVal(real_t val) {
    leaf_val = val;
  }
  // Left child
  inline DTNode* LeftChild() {
    return l_child;
  }
  inline void SetLeftChild(DTNode* node) {
    l_child = node;
  }
  // Right child
  inline DTNode* RightChild() {
    return r_child;
  }
  inline void SetRightChild(DTNode* node) {
    r_child = node;
  }
  // Best feature id
  inline uint32_t BestFeatID() {
    return best_feat_id;
  }
  inline void SetBestFeatID(uint32_t id) {
    best_feat_id = id;
  }
  // Best bin value
  inline uint32_t BestBinVal() {
    return best_bin_val;
  }
  inline void SetBestBinVal(uint8_t val) {
    best_bin_val = val;
  }
  // Left or Right node?
  inline char LeftOrRight() {
    return info->l_or_r;
  }
  inline void SetLeftOrRight(char ch) {
    info->l_or_r = ch;
  }
  // Node level
  inline uint32_t Level() {
    return info->level;
  }
  inline void SetLevel(uint32_t level) {
    info->level = level;
  }
  // Start postion
  inline uint32_t StartPos() {
    return info->start_pos;
  }
  inline void SetStartPos(uint32_t pos) {
    info->start_pos = pos;
  }
  // End position
  inline uint32_t EndPos() {
    return info->end_pos;
  }
  inline void SetEndPos(uint32_t pos) {
    info->end_pos = pos;
  }
  // Split position
  inline uint32_t MidPos() {
    return info->mid_pos;
  }
  inline void SetMidPos(uint32_t pos) {
    info->mid_pos = pos;
  }
  // Best gini
  inline real_t BestGini() {
    return info->best_gini;
  }
  inline void SetBestGini(real_t gini) {
    info->best_gini = gini;
  } 
  // Parent node
  inline DTNode* Parent() {
    return info->parent;
  }
  inline void SetParent(DTNode* node) {
    info->parent = node;
  }
  // Brother node
  inline DTNode* Brother() {
    return info->brother;
  }
  inline void SetBrother(DTNode* node) {
    info->brother = node;
  }
  // Histogram
  inline Histogram* Histo() {
    return info->histo;
  }
  inline void SetHisto(Histogram* gram) {
    info->histo = gram;
  }
  // Data size
  inline uint32_t DataSize() {
    return info->end_pos-info->start_pos+1;
  }
};

//-----------------------------------------------------------------------------
// Decision tree data structure
//-----------------------------------------------------------------------------
struct DTree {
  // size of leaf
  uint32_t leaf_size = 1;
  // Depth of Tree
  uint32_t max_depth = 1;
  // root node
  struct DTNode* root = nullptr;
  // Row sample
  std::vector<uint32_t> rowIdx;
  // Column sample
  std::vector<uint32_t> colIdx;
  // Bootstrap sample for data row
  void SampleRow() {
    rowIdx.resize(NUM_DATA);
    for (size_t i = 0; i < NUM_DATA; ++i) {
      rowIdx[i] = i;
    }
  }
  // Column sampling
  void SampleCol() {
    colIdx.resize(NUM_FEAT);
    for (size_t i = 0; i < NUM_FEAT; ++i) {
      colIdx[i] = i;
    }
  }
};

//-----------------------------------------------------------------------------
// Find the best split position for current node
//-----------------------------------------------------------------------------
static void FindPosition(DTNode* node, 
                         DTree* tree, 
                         uint8_t* X, 
                         real_t* Y) {
  Timer timer;
  timer.tic();
  uint32_t len = node->DataSize();
  uint32_t start_pos = node->StartPos();
  uint32_t end_pos = node->EndPos();
  uint32_t* count = node->Histo()->count;
  std::vector<uint32_t>& row = tree->rowIdx;
  std::vector<uint32_t>& col = tree->colIdx;
  uint32_t col_size = col.size();
  uint32_t cc = num_class * col_size;
  // Collect histogram
  if (node->LeftOrRight() == 'l' || 
      node->Brother()->IsLeaf()) { 
    for (uint32_t i = start_pos; i <= end_pos; ++i) {
      uint32_t row_idx = row[i];
      int y = Y[row_idx];
      uint8_t* ptr = X + row_idx * NUM_FEAT;
      for (uint32_t j = 0; j < col_size; ++j) {
        uint8_t tmp = *(ptr + j);
        uint32_t index = num_class * (tmp * col_size + j) + y;
        count[index]++;
      }
    }
  } else {  // right_histo = parent_histo - left_histo
    uint32_t* count_parent = node->Parent()->Histo()->count;
    uint32_t* count_brother = node->Brother()->Histo()->count;
    uint32_t count_len = node->Histo()->count_len;
    for (uint32_t i = 0; i < count_len; ++i) {
      count[i] = count_parent[i] - count_brother[i];
    }
  }
  // Sum total count via the first col histo
  std::vector<uint32_t> total_count(num_class, 0);
  for (uint32_t i = 0; i <= max_bin; ++i) {
    uint32_t* ptr = count + i*cc;
    for (uint8_t c = 0; c < num_class; ++c) {
      total_count[c] += *ptr;
      ptr++;
    }
  }
  // Find best split position
  for (uint32_t j = 0; j < col_size; ++j) {
    std::vector<uint32_t> left_count(num_class, 0);
    std::vector<uint32_t> right_count(total_count);
    uint32_t* base_ptr = count + j*num_class;
    for (uint32_t i = 0; i <= max_bin; ++i) {
      uint32_t* ptr = base_ptr + cc*i;
      for (uint8_t c = 0; c < num_class; ++c) {
        left_count[c] += *ptr;
        right_count[c] -= *ptr;
        ptr++;
      }
      uint32_t left_sum = std::accumulate(
        left_count.begin(), left_count.end(), 0);
      uint32_t right_sum = std::accumulate(
        right_count.begin(), right_count.end(), 0);
      real_t real_left_sum = 0.0;
      real_t real_right_sum = 0.0;
      for (uint8_t c = 0; c < num_class; ++c) {
        real_t tmp = (real_t)left_count[c] / left_sum;
        real_left_sum += tmp*tmp;
        tmp = (real_t)right_count[c] / right_sum;
        real_right_sum += tmp*tmp;
      }
      real_t left_gini = 1.0 - real_left_sum;
      left_gini *= (real_t)left_sum / len;
      real_t right_gini = 1.0 - real_right_sum;
      right_gini *= (real_t)right_sum / len;
      real_t gini = left_gini + right_gini;
      // Find updated gini
      if (gini < node->BestGini()) {
        node->SetBestGini(gini);
        node->SetBestFeatID(col[j]);
        node->SetBestBinVal(i);
      }
    }
  }
  if (node->LeftOrRight() == 'r') {
    node->ClearParent();
  }
  time_find += timer.toc();
}

//-----------------------------------------------------------------------------
// Split Data
//-----------------------------------------------------------------------------
static void SplitData(DTNode* node, 
                      DTree* tree, 
                      uint8_t* X,
                      real_t* Y) {
  Timer timer;
  timer.tic();
  std::vector<uint32_t>& idx = tree->rowIdx;
  uint32_t ptr_head = node->StartPos();
  uint32_t ptr_tail = node->EndPos();
  uint32_t best_feat_id = node->BestFeatID();
  uint8_t best_bin_val = node->BestBinVal();
  uint8_t* ptr = X + best_feat_id;
  while (ptr_head < ptr_tail) {
    uint8_t bin = *(ptr + idx[ptr_head] * NUM_FEAT);
    if (bin <= best_bin_val) {
      ptr_head++;
    } else {
      // swap head and tail
      idx[ptr_head] ^= idx[ptr_tail];
      idx[ptr_tail] ^= idx[ptr_head];
      idx[ptr_head] ^= idx[ptr_tail];
      ptr_tail--;
    }
  }
  node->SetMidPos(ptr_head - 1);
  time_split += timer.toc();
}

//-----------------------------------------------------------------------------
// Calculate the leaf value
//-----------------------------------------------------------------------------
static real_t leaf(DTNode* node, 
                  DTree* tree, 
                  real_t* Y) {
  std::vector<uint32_t> count(num_class, 0);
  std::vector<uint32_t>::iterator result;
  uint32_t start_pos = node->StartPos();
  uint32_t end_pos = node->EndPos();
  std::vector<uint32_t>& idx = tree->rowIdx;
  for (uint32_t i = start_pos; i <= end_pos; ++i) {
    count[(int)Y[idx[i]]]++;
  }
  result = std::max_element(count.begin(), count.end());
  return (real_t)std::distance(count.begin(), result);
}

//-----------------------------------------------------------------------------
// Return true and set the leaf node value if 
// current node is leaf node
//-----------------------------------------------------------------------------
static bool IsLeaf(DTNode* node, 
                   DTree* tree, 
                   real_t* Y) {
  // If is leaf node
  if (node->Level() == MAX_DEPTH || 
      node->DataSize() < MIN_SPLIT) {
    node->SetLeaf();
    node->SetLeafVal(leaf(node, tree, Y));
    // clear tmp info in node
    node->Clear();
    return true;
  }
  return false;
}

//-----------------------------------------------------------------------------
// Build Decision Tree
//-----------------------------------------------------------------------------
static void BuildDTree(uint8_t* X, real_t* Y, DTree* tree) {
  DTNode* root = new DTNode();
  root->Init(tree->colIdx.size(), max_bin+1, num_class);
  // treat root as left node
  root->SetLeftOrRight('l');
  root->SetLevel(1);
  root->SetStartPos(0);
  root->SetEndPos(NUM_DATA-1);
  tree->root = root;
  // Queue for BFS traverse
  std::queue<DTNode*> node_queue;
  node_queue.push(root);
  while (!node_queue.empty()) {
    DTNode *node = node_queue.front();
    if (IsLeaf(node, tree, Y) == false) {
      FindPosition(node, tree, X, Y);
      SplitData(node, tree, X, Y);
      // New left child
      DTNode *l_node = new DTNode();
      l_node->Init(tree->colIdx.size(), max_bin+1, num_class);
      l_node->SetLeftOrRight('l');
      l_node->SetStartPos(node->StartPos());
      l_node->SetEndPos(node->MidPos());
      l_node->SetLevel(node->Level() + 1);
      // New right child
      DTNode *r_node = new DTNode();
      r_node->Init(tree->colIdx.size(), max_bin+1, num_class);
      r_node->SetLeftOrRight('r');
      r_node->SetStartPos(node->MidPos() + 1);
      r_node->SetEndPos(node->EndPos());
      r_node->SetLevel(node->Level() + 1);
      // Right node can use parent and 
      // brother to calculate histogram bin value
      r_node->SetParent(node);
      r_node->SetBrother(l_node);
      // Push new node
      node->SetLeftChild(l_node);
      node->SetRightChild(r_node);
      node_queue.push(l_node);
      node_queue.push(r_node);
      // Update tree level and leaf size
      if (r_node->Level() > tree->max_depth) {
        tree->max_depth = r_node->Level();
      }
      tree->leaf_size++;
    }
    node_queue.pop();
  }
}

//-----------------------------------------------------------------------------
// Get the leaf node by given the X
//-----------------------------------------------------------------------------
static DTNode* getleaf(DTNode* node, uint8_t* x) {
  if (node->IsLeaf()) {
    return node;
  }
  uint32_t id = node->BestFeatID();
  uint8_t val = node->BestBinVal();
  if (x[id] <= val) {
    return getleaf(node->LeftChild(), x);
  } else {
    return getleaf(node->RightChild(), x);
  }
}

//-----------------------------------------------------------------------------
// Prediction method
//-----------------------------------------------------------------------------
static void Predict(DTree* tree, uint8_t* x, real_t* y) {
  for (uint32_t i = 0; i < NUM_TEST_DATA; ++i) {
    DTNode* leaf = getleaf(tree->root, x);
    y[i] = leaf->LeafVal();
    x += NUM_FEAT;
  }
}

//-----------------------------------------------------------------------------
// Calculate the accuracy
//-----------------------------------------------------------------------------
static real_t accuracy(real_t* pred_Y, real_t* y, size_t n) {
  uint32_t k = 0;
  for (uint32_t i = 0; i < n; ++i) {
    if (pred_Y[i] == y[i]) {
      ++k;
    }
  }
  return (real_t) k / (real_t) n;
}

int main(int argc, char** argv) {
//-----------------------------------------------------------------------------
// Read data
//-----------------------------------------------------------------------------
  Timer timer;
  timer.tic();
  real_t *Y = new real_t[NUM_DATA];
  real_t *X = new real_t[NUM_DATA * NUM_FEAT];
  read_data(tr_file, X, Y, NUM_DATA); 
  float time = timer.toc();
  std::cout << "Read time: " << time << " (s)\n"; 
//-----------------------------------------------------------------------------
// Find max_feat and min_feat (this process can use sample)
//-----------------------------------------------------------------------------
  timer.reset();
  timer.tic();
  std::vector<MaxMin> MaxMinVec(NUM_FEAT);
  for (size_t i = 0; i < NUM_DATA; ++i) {
    for (size_t j = 0; j < NUM_FEAT; ++j) {
      real_t feat = X[i*NUM_FEAT+j];
      if (feat > MaxMinVec[j].max_feat) {
        MaxMinVec[j].max_feat = feat;
      } else if (feat < MaxMinVec[j].min_feat) {
        MaxMinVec[j].min_feat = feat;
      }
    }
  }
  for (size_t i = 0; i < NUM_FEAT; ++i) {
    MaxMinVec[i].gap = 
      (MaxMinVec[i].max_feat - MaxMinVec[i].min_feat) / max_bin;
  }
  time = timer.toc();
  std::cout << "Find MaxMin time: " << time << " (s)\n";
//-----------------------------------------------------------------------------
// Re-build X
//-----------------------------------------------------------------------------
  timer.reset();
  timer.tic();
  uint8_t* Matrix = new uint8_t[NUM_DATA * NUM_FEAT];
  for (size_t i = 0; i < NUM_DATA; ++i) {
    for (size_t j = 0; j < NUM_FEAT; ++j) {
      size_t idx = i*NUM_FEAT+j;
      real_t tmp = X[idx] - MaxMinVec[j].min_feat;
      Matrix[idx] = uint8_t(floor(tmp / MaxMinVec[j].gap));
      if (Matrix[idx] > max_bin) {
        Matrix[idx] = max_bin;
      } else if (Matrix[idx] < 0) {
        Matrix[idx] = 0;
      }
    }
  }  
  time = timer.toc();
  std::cout << "Build Matrix time: " << time << " (s)\n";
//-----------------------------------------------------------------------------
// Build Tree
//-----------------------------------------------------------------------------
  real_t*  Data_Y = Y;
  uint8_t* Data_X = Matrix;
  DTree *tree = new DTree();
  tree->SampleRow();
  tree->SampleCol();
  timer.reset();
  timer.tic();
  BuildDTree(Data_X, Data_Y, tree);
  time = timer.toc();
  std::cout << "Tree depth: " << tree->max_depth << std::endl;
  std::cout << "Leaf size: " << tree->leaf_size << std::endl;
  std::cout << "xforest build time: " << time << " (s)\n";
  std::cout << "  Find time: " << time_find << " (s)\n";
  std::cout << "  Split time: " << time_split << " (s)\n";
//-----------------------------------------------------------------------------
// Read test data
//-----------------------------------------------------------------------------
  timer.reset();
  timer.tic();
  real_t *test_Y = new real_t[NUM_TEST_DATA];
  real_t *pred_Y = new real_t[NUM_TEST_DATA];
  real_t *test_X = new real_t[NUM_TEST_DATA * NUM_FEAT];
  read_data(te_file, test_X, test_Y, NUM_TEST_DATA); 
  time = timer.toc();
  std::cout << "Read test time: " << time << " (s)\n"; 
//-----------------------------------------------------------------------------
// Re-build test_X
//-----------------------------------------------------------------------------
  timer.reset();
  timer.tic();
  uint8_t* test_Matrix = new uint8_t[NUM_TEST_DATA * NUM_FEAT];
  for (size_t i = 0; i < NUM_TEST_DATA; ++i) {
    for (size_t j = 0; j < NUM_FEAT; ++j) {
      size_t idx = i*NUM_FEAT+j;
      real_t tmp = test_X[idx] - MaxMinVec[j].min_feat;
      test_Matrix[idx] = uint8_t(floor(tmp / MaxMinVec[j].gap));
      if (test_Matrix[idx] > max_bin) {
        test_Matrix[idx] = max_bin;
      } else if (test_Matrix[idx] < 0) {
        test_Matrix[idx] = 0;
      }
    }
  }  
  time = timer.toc();
  std::cout << "Build test_X time: " << time << " (s)\n";
//-----------------------------------------------------------------------------
// Make Prediction
//-----------------------------------------------------------------------------
  timer.reset();
  timer.tic();
  Predict(tree, test_Matrix, pred_Y);
  real_t acc = accuracy(pred_Y, test_Y, NUM_TEST_DATA);
  std::cout << "accuracy: " << acc << std::endl;
  time = timer.toc();
  std::cout << "Inference time: " << time << " (s)\n"; 
  return 0;
}
