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

// DCT and related functions
typedef struct {
  PyObject_HEAD
  boost::shared_ptr<bob::sp::DCT1D> cxx;
} PyBobSpDCT1DObject;
extern PyTypeObject PyBobSpDCT1D_Type;
bool init_BobSpDCT1D(PyObject* module);

typedef struct {
  PyObject_HEAD
  boost::shared_ptr<bob::sp::DCT2D> cxx;
} PyBobSpDCT2DObject;
extern PyTypeObject PyBobSpDCT2D_Type;
bool init_BobSpDCT2D(PyObject* module);

PyObject* PyBobSpDCT(PyObject*, PyObject* args, PyObject* kwds);
extern bob::extension::FunctionDoc s_dct;

PyObject* PyBobSpIDCT(PyObject*, PyObject* args, PyObject* kwds);
extern bob::extension::FunctionDoc s_idct;



#endif // BOB_SP_MAIN_H
