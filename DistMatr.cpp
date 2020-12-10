// [[Rcpp::depends(RcppParallel)]]
#include <Rcpp.h>
#include <RcppParallel.h>
#include <cmath>
#include <algorithm>
using namespace Rcpp;
using namespace RcppParallel;

// generic function euclidean distance between vectors
template <typename InputIterator1, typename InputIterator2>
inline double euclid_dist(InputIterator1 begin1, InputIterator1 end1, 
                            InputIterator2 begin2) {
  
  // value to return
  double rval = 0;
  
  // set iterators to beginning of ranges
  InputIterator1 it1 = begin1;
  InputIterator2 it2 = begin2;
  
  // for each input item
  while (it1 != end1) {
    
    // euclidean distance
    double dist = (*it1++) - (*it2++);
    rval += dist * dist;

  }
  return rval > 0.0 ? sqrt(rval) : 0.0;
}

struct distanceMatr : public Worker { //function object
  
  // input matrix
  const RMatrix<double> amat;
  const RMatrix<double> bmat;
  
  // output matrix to write to
  RMatrix<double> rmat;
  
  // initialize from Rcpp input and output matrixes
  distanceMatr(const NumericMatrix amat, const NumericMatrix bmat, NumericMatrix rmat)
    : amat(amat), bmat(bmat), rmat(rmat) {}
  
  // function call operator that work for the specified range (begin/end) 
  void operator()(std::size_t begin, std::size_t end) {
    for (std::size_t i = begin; i < end; i++) {
      
      for (std::size_t j = 0; j < bmat.nrow(); j++) {
        
        // rows we will operate on
        RcppParallel::RMatrix<double>::Row row1 = amat.row(i);
        RcppParallel::RMatrix<double>::Row row2 = bmat.row(j);
        
        // write to output matrix
        rmat(i,j) = euclid_dist(row1.begin(), row1.end(), row2.begin()); //Euclidean distance between rows by calling generic euclidean distance function for each row combination
        
      }
    }
  }
};

// [[Rcpp::export]]
NumericMatrix distanceMatr_parallel(NumericMatrix amat, NumericMatrix bmat) {
  
  // allocate the matrix we will return
  NumericMatrix rmat(amat.nrow(), bmat.nrow());
  
  // create the worker
  distanceMatr distanceMatr(amat, bmat, rmat);
  
  // call it with parallelFor
  parallelFor(0, amat.nrow(), distanceMatr);
  
  return rmat;
}