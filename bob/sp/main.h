/**
 * @author Manuel Gunther <siebenkopf@googlemail.com>
 * @date Tue Dec  8 10:06:29 MST 2015
 *
 * @brief Bindings to bob::sp
 */

#ifndef BOB_SP_MAIN_H
#define BOB_SP_MAIN_H

#include <bob.blitz/cppapi.h>
#include <bob.blitz/cleanup.h>
#include <bob.core/api.h>
#include <bob.extension/documentation.h>

#include <bob.sp/DCT1D.h>
#include <bob.sp/DCT2D.h>
#include <bob.sp/FFT1D.h>
#include <bob.sp/FFT2D.h>
#include <bob.sp/extrapolate.h>
#include <bob.sp/Quantization.h>


// DCT and related functions
typedef struct {
  PyObject_HEAD
  boost::shared_ptr<bob::sp::DCT1D> cxx;
} PyBobSpDCT1DObject;
extern PyTypeObject PyBobSpDCT1D_Type;
bool init_BobSpDCT1D(PyObject* module);

typedef struct {
  PyObject_HEAD
  boost::shared_ptr<bob::sp::IDCT1D> cxx;
} PyBobSpIDCT1DObject;
extern PyTypeObject PyBobSpIDCT1D_Type;
bool init_BobSpIDCT1D(PyObject* module);

typedef struct {
  PyObject_HEAD
  boost::shared_ptr<bob::sp::DCT2D> cxx;
} PyBobSpDCT2DObject;
extern PyTypeObject PyBobSpDCT2D_Type;
bool init_BobSpDCT2D(PyObject* module);

typedef struct {
  PyObject_HEAD
  boost::shared_ptr<bob::sp::IDCT2D> cxx;
} PyBobSpIDCT2DObject;
extern PyTypeObject PyBobSpIDCT2D_Type;
bool init_BobSpIDCT2D(PyObject* module);


PyObject* PyBobSpDCT(PyObject*, PyObject* args, PyObject* kwds);
extern bob::extension::FunctionDoc s_dct;

PyObject* PyBobSpIDCT(PyObject*, PyObject* args, PyObject* kwds);
extern bob::extension::FunctionDoc s_idct;


// FFT and related functions
typedef struct {
  PyObject_HEAD
  boost::shared_ptr<bob::sp::FFT1D> cxx;
} PyBobSpFFT1DObject;
extern PyTypeObject PyBobSpFFT1D_Type;
bool init_BobSpFFT1D(PyObject* module);

typedef struct {
  PyObject_HEAD
  boost::shared_ptr<bob::sp::IFFT1D> cxx;
} PyBobSpIFFT1DObject;
extern PyTypeObject PyBobSpIFFT1D_Type;
bool init_BobSpIFFT1D(PyObject* module);

typedef struct {
  PyObject_HEAD
  boost::shared_ptr<bob::sp::FFT2D> cxx;
} PyBobSpFFT2DObject;
extern PyTypeObject PyBobSpFFT2D_Type;
bool init_BobSpFFT2D(PyObject* module);

typedef struct {
  PyObject_HEAD
  boost::shared_ptr<bob::sp::IFFT2D> cxx;
} PyBobSpIFFT2DObject;
extern PyTypeObject PyBobSpIFFT2D_Type;
bool init_BobSpIFFT2D(PyObject* module);


PyObject* PyBobSpFFT(PyObject*, PyObject* args, PyObject* kwds);
extern bob::extension::FunctionDoc s_fft;

PyObject* PyBobSpIFFT(PyObject*, PyObject* args, PyObject* kwds);
extern bob::extension::FunctionDoc s_ifft;

PyObject* PyBobSpFFTShift(PyObject*, PyObject* args, PyObject* kwds);
extern bob::extension::FunctionDoc s_fftshift;

PyObject* PyBobSpIFFTShift(PyObject*, PyObject* args, PyObject* kwds);
extern bob::extension::FunctionDoc s_ifftshift;


// extrapolation
bool init_BobSpBorderType(PyObject* module);
PyObject* PyBobSpExtrapolate(PyObject*, PyObject* args, PyObject* kwds);
extern bob::extension::FunctionDoc s_extrapolate;


// Quantization
typedef struct {
  PyObject_HEAD
  int type_num;
  boost::shared_ptr<void> cxx;
} PyBobSpQuantizationObject;
extern PyTypeObject PyBobSpQuantization_Type;
bool init_BobSpQuantization(PyObject* module);


#endif // BOB_SP_MAIN_H
