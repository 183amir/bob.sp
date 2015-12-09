/**
 * @date Mon Jun 20 11:47:58 2011 +0200
 * @author Andre Anjos <andre.anjos@idiap.ch>
 *
 * @brief Python bindings to statistical methods
 *
 * Copyright (C) 2011-2013 Idiap Research Institute, Martigny, Switzerland
 */

#include <bob.blitz/cppapi.h>
#include <bob.blitz/cleanup.h>
#include <bob.extension/defines.h>
#include <bob.sp/FFT1D.h>


#include "main.h"

static auto s_fft1d = bob::extension::ClassDoc(
  BOB_EXT_MODULE_PREFIX ".FFT1D",
  "Calculates the direct FFT of a 1D array/signal",
  "Input and output arrays are 1D NumPy arrays of type ``complex128``."
)
.add_constructor(bob::extension::FunctionDoc(
  "FFT1D",
  "Creates a new FFT1D extractor in the given dimensions"
)
.add_prototype("length", "")
.add_prototype("other", "")
.add_parameter("length", "int", "The length of the input signal this class will be able to handle")
.add_parameter("other", ":py:class:`FFT1D`", "The other FFT1D class to copy-construct")
);

PyTypeObject PyBobSpFFT1D_Type = {
  PyVarObject_HEAD_INIT(0, 0)
  0
};

int PyBobSpFFT1D_Check(PyObject* o) {
  return PyObject_IsInstance(o, reinterpret_cast<PyObject*>(&PyBobSpFFT1D_Type));
}

static void PyBobSpFFT1D_Delete (PyBobSpFFT1DObject* self) {
  self->cxx.reset();
  Py_TYPE(self)->tp_free((PyObject*)self);
}

static int PyBobSpFFT1D_InitCopy(PyBobSpFFT1DObject* self, PyObject* args, PyObject* kwds) {
  char** kwlist = s_fft1d.kwlist(1);

  PyBobSpFFT1DObject* other;
  if (!PyArg_ParseTupleAndKeywords(args, kwds, "O!", kwlist, &PyBobSpFFT1D_Type, &other)) return -1;

  self->cxx.reset(new bob::sp::FFT1D(*(other->cxx)));
  return 0;
}

static int PyBobSpFFT1D_InitShape(PyBobSpFFT1DObject* self, PyObject *args, PyObject* kwds) {
  char** kwlist = s_fft1d.kwlist(0);

  Py_ssize_t length = 0;
  if (!PyArg_ParseTupleAndKeywords(args, kwds, "n", kwlist, &length)) return -1;

  self->cxx.reset(new bob::sp::FFT1D(length));
  return 0; ///< SUCCESS
}

static int PyBobSpFFT1D_Init(PyBobSpFFT1DObject* self, PyObject* args, PyObject* kwds) {
BOB_TRY
  Py_ssize_t nargs = (args?PyTuple_Size(args):0) + (kwds?PyDict_Size(kwds):0);

  switch (nargs) {
    case 1:
      {
        PyObject* arg = 0; ///< borrowed (don't delete)
        if (PyTuple_Size(args)) arg = PyTuple_GET_ITEM(args, 0);
        else {
          PyObject* tmp = PyDict_Values(kwds);
          auto tmp_ = make_safe(tmp);
          arg = PyList_GET_ITEM(tmp, 0);
        }

        if (PyBob_NumberCheck(arg))
          return PyBobSpFFT1D_InitShape(self, args, kwds);

        if (PyBobSpFFT1D_Check(arg))
          return PyBobSpFFT1D_InitCopy(self, args, kwds);

        PyErr_Format(PyExc_TypeError, "cannot initialize `%s' with `%s' (see help)", Py_TYPE(self)->tp_name, Py_TYPE(arg)->tp_name);
      }
      break;

    default:
      PyErr_Format(PyExc_RuntimeError, "number of arguments mismatch - %s requires 1 argument, but you provided %" PY_FORMAT_SIZE_T "d (see help)", Py_TYPE(self)->tp_name, nargs);
  }
  return -1;
BOB_CATCH_MEMBER("constructor", -1)
}

static PyObject* PyBobSpFFT1D_Repr(PyBobSpFFT1DObject* self) {
BOB_TRY
  return PyString_FromFormat("%s(length=%zu)", Py_TYPE(self)->tp_name, self->cxx->getLength());
BOB_CATCH_MEMBER("__repr__", 0)
}

static PyObject* PyBobSpFFT1D_RichCompare (PyBobSpFFT1DObject* self, PyObject* other, int op) {
BOB_TRY
  if (!PyBobSpFFT1D_Check(other)) {
    PyErr_Format(PyExc_TypeError, "cannot compare `%s' with `%s'",
        Py_TYPE(self)->tp_name, Py_TYPE(other)->tp_name);
    return 0;
  }

  auto other_ = reinterpret_cast<PyBobSpFFT1DObject*>(other);

  switch (op) {
    case Py_EQ:
      if (self->cxx->operator==(*other_->cxx)) Py_RETURN_TRUE;
      Py_RETURN_FALSE;
      break;
    case Py_NE:
      if (self->cxx->operator!=(*other_->cxx)) Py_RETURN_TRUE;
      Py_RETURN_FALSE;
      break;
    default:
      Py_INCREF(Py_NotImplemented);
      return Py_NotImplemented;
  }
BOB_CATCH_MEMBER("RichCompare", 0)
}

static auto s_length = bob::extension::VariableDoc(
  "length",
  "int",
  "The length of the input and output vector"
);
static PyObject* PyBobSpFFT1D_GetLength(PyBobSpFFT1DObject* self, void* /*closure*/) {
BOB_TRY
  return Py_BuildValue("n", self->cxx->getLength());
BOB_CATCH_MEMBER("length", 0)
}

static int PyBobSpFFT1D_SetLength(PyBobSpFFT1DObject* self, PyObject* o, void* /*closure*/) {
BOB_TRY
  if (!PyBob_NumberCheck(o)) {
    PyErr_Format(PyExc_TypeError, "`%s' length can only be set using a number, not `%s'", Py_TYPE(self)->tp_name, Py_TYPE(o)->tp_name);
    return -1;
  }

  Py_ssize_t len = PyNumber_AsSsize_t(o, PyExc_OverflowError);
  if (PyErr_Occurred()) return -1;

  self->cxx->setLength(len);
  return 0;
BOB_CATCH_MEMBER("length", -1)
}

static auto s_shape = bob::extension::VariableDoc(
  "shape",
  "(int)",
  "A 1D tuple that represents the size of the input/output vector"
);
static PyObject* PyBobSpFFT1D_GetShape(PyBobSpFFT1DObject* self, void* /*closure*/) {
BOB_TRY
  return Py_BuildValue("(n)", self->cxx->getLength());
BOB_CATCH_MEMBER("shape", 0)
}

static int PyBobSpFFT1D_SetShape(PyBobSpFFT1DObject* self, PyObject* o, void* /*closure*/) {
BOB_TRY
  if (!PySequence_Check(o)) {
    PyErr_Format(PyExc_TypeError, "`%s' shape can only be set using tuples (or sequences), not `%s'", Py_TYPE(self)->tp_name, Py_TYPE(o)->tp_name);
    return -1;
  }

  PyObject* shape = PySequence_Tuple(o);
  auto shape_ = make_safe(shape);

  if (PyTuple_GET_SIZE(shape) != 1) {
    PyErr_Format(PyExc_RuntimeError, "`%s' shape can only be set using 1-position tuples (or sequences), not an %" PY_FORMAT_SIZE_T "d-position sequence", Py_TYPE(self)->tp_name, PyTuple_GET_SIZE(shape));
    return -1;
  }

  Py_ssize_t len = PyNumber_AsSsize_t(PyTuple_GET_ITEM(shape, 0), PyExc_OverflowError);
  if (PyErr_Occurred()) return -1;

  self->cxx->setLength(len);
  return 0;
BOB_CATCH_MEMBER("shape", -1)
}

static PyGetSetDef PyBobSpFFT1D_getseters[] = {
    {
      s_length.name(),
      (getter)PyBobSpFFT1D_GetLength,
      (setter)PyBobSpFFT1D_SetLength,
      s_length.doc(),
      0
    },
    {
      s_shape.name(),
      (getter)PyBobSpFFT1D_GetShape,
      (setter)PyBobSpFFT1D_SetShape,
      s_shape.doc(),
      0
    },
    {0}  /* Sentinel */
};


static auto s_transform = bob::extension::FunctionDoc(
  "transform",
  "Calculates the direct FFT of the given signal",
  ".. note:: The :py:meth:`__call__` function is a synonym for this function.",
  true
)
.add_prototype("input, [output]", "output")
.add_parameter("input", "array_like(1D, complex)", "The given input array. It must be of length :py:attr:`length`")
.add_parameter("output", "array_like(1D, complex)", "A pre-allocated output array. If given, it must be of length :py:attr:`length`")
.add_return("output", "array_like(1D, complex)", "The FFT result; identical to the ``output`` parameter, if given")
;
static PyObject* PyBobSpFFT1D_transform(PyBobSpFFT1DObject* self, PyObject* args, PyObject* kwds) {
BOB_TRY
  char** kwlist = s_transform.kwlist();

  PyBlitzArrayObject* input;
  PyBlitzArrayObject* output = 0;

  if (!PyArg_ParseTupleAndKeywords(args, kwds, "O&|O&", kwlist,
        &PyBlitzArray_Converter, &input,
        &PyBlitzArray_OutputConverter, &output
        )) return 0;

  //protects acquired resources through this scope
  auto input_ = make_safe(input);
  auto output_ = make_xsafe(output);

  if (input->type_num != NPY_COMPLEX128) {
    PyErr_Format(PyExc_TypeError, "`%s' only supports 128-bit complex (2x64-bit float) arrays for input array `input'", Py_TYPE(self)->tp_name);
    return 0;
  }

  if (output && output->type_num != NPY_COMPLEX128) {
    PyErr_Format(PyExc_TypeError, "`%s' only supports 128-bit complex (2x64-bit float) arrays for output array `output'", Py_TYPE(self)->tp_name);
    return 0;
  }

  if (input->ndim != 1) {
    PyErr_Format(PyExc_TypeError, "`%s' only accepts 1-dimensional arrays (not %" PY_FORMAT_SIZE_T "dD arrays)", Py_TYPE(self)->tp_name, input->ndim);
    return 0;
  }

  if (output && input->ndim != output->ndim) {
    PyErr_Format(PyExc_RuntimeError, "Input and output arrays should have matching number of dimensions, but input array `input' has %" PY_FORMAT_SIZE_T "d dimensions while output array `output' has %" PY_FORMAT_SIZE_T "d dimensions", input->ndim, output->ndim);
    return 0;
  }

  if (output && output->shape[0] != (Py_ssize_t)self->cxx->getLength()) {
    PyErr_Format(PyExc_RuntimeError, "1D `output' array should have %" PY_FORMAT_SIZE_T "d elements matching `%s' output size, not %" PY_FORMAT_SIZE_T "d elements", self->cxx->getLength(), Py_TYPE(self)->tp_name, output->shape[0]);
    return 0;
  }

  /** if ``output`` was not pre-allocated, do it now **/
  if (!output) {
    Py_ssize_t length = self->cxx->getLength();
    output = (PyBlitzArrayObject*)PyBlitzArray_SimpleNew(NPY_COMPLEX128, 1, &length);
    output_ = make_safe(output);
  }

  /** all basic checks are done, can call the operator now **/
  self->cxx->operator()(*PyBlitzArrayCxx_AsBlitz<std::complex<double>,1>(input), *PyBlitzArrayCxx_AsBlitz<std::complex<double>,1>(output));

  return PyBlitzArray_AsNumpyArray(output, 0);
BOB_CATCH_MEMBER("transform", 0)
}

static PyMethodDef PyBobSpFFT1D_methods[] = {
    {
      s_transform.name(),
      (PyCFunction)PyBobSpFFT1D_transform,
      METH_VARARGS|METH_KEYWORDS,
      s_transform.doc()
    },
    {0}  /* Sentinel */
};

bool init_BobSpFFT1D(PyObject* module){
  // class definition
  PyBobSpFFT1D_Type.tp_name = s_fft1d.name();
  PyBobSpFFT1D_Type.tp_basicsize = sizeof(PyBobSpFFT1DObject);
  PyBobSpFFT1D_Type.tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE;
  PyBobSpFFT1D_Type.tp_doc = s_fft1d.doc();

  PyBobSpFFT1D_Type.tp_new = PyType_GenericNew;
  PyBobSpFFT1D_Type.tp_dealloc = reinterpret_cast<destructor>(PyBobSpFFT1D_Delete);
  PyBobSpFFT1D_Type.tp_init = reinterpret_cast<initproc>(PyBobSpFFT1D_Init);
  PyBobSpFFT1D_Type.tp_richcompare = reinterpret_cast<richcmpfunc>(PyBobSpFFT1D_RichCompare);
  PyBobSpFFT1D_Type.tp_methods = PyBobSpFFT1D_methods;
  PyBobSpFFT1D_Type.tp_getset = PyBobSpFFT1D_getseters;
  PyBobSpFFT1D_Type.tp_call = reinterpret_cast<ternaryfunc>(PyBobSpFFT1D_transform);
  PyBobSpFFT1D_Type.tp_repr = reinterpret_cast<reprfunc>(PyBobSpFFT1D_Repr);

  // check that everything is fine
  if (PyType_Ready(&PyBobSpFFT1D_Type) < 0) return false;
  return PyModule_AddObject(module, "FFT1D", Py_BuildValue("O", &PyBobSpFFT1D_Type)) >= 0;
}
