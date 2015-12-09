/**
 * @date Mon Jun 20 11:47:58 2011 +0200
 * @author Andre Anjos <andre.anjos@idiap.ch>
 *
 * @brief Python bindings to statistical methods
 *
 * Copyright (C) 2011-2013 Idiap Research Institute, Martigny, Switzerland
 */

#include "main.h"

static auto s_idct1d = bob::extension::ClassDoc(
  BOB_EXT_MODULE_PREFIX ".IDCT1D",
  "Calculates the inverse DCT of a 1D array/signal",
  "Input and output arrays are 1D NumPy arrays of type ``float64``."
)
.add_constructor(bob::extension::FunctionDoc(
  "IDCT1D",
  "Creates a new IDCT1D extractor in the given dimensions"
)
.add_prototype("length", "")
.add_prototype("other", "")
.add_parameter("length", "int", "The length of the input signal this class will be able to handle")
.add_parameter("other", ":py:class:`IDCT1D`", "The other IDCT1D class to copy-construct")
);

PyTypeObject PyBobSpIDCT1D_Type = {
  PyVarObject_HEAD_INIT(0, 0)
  0
};

int PyBobSpIDCT1D_Check(PyObject* o) {
  return PyObject_IsInstance(o, reinterpret_cast<PyObject*>(&PyBobSpIDCT1D_Type));
}

static void PyBobSpIDCT1D_Delete (PyBobSpIDCT1DObject* self) {
  self->cxx.reset();
  Py_TYPE(self)->tp_free((PyObject*)self);
}

static int PyBobSpIDCT1D_InitCopy(PyBobSpIDCT1DObject* self, PyObject* args, PyObject* kwds) {
  char** kwlist = s_idct1d.kwlist(1);

  PyBobSpIDCT1DObject* other;
  if (!PyArg_ParseTupleAndKeywords(args, kwds, "O!", kwlist, &PyBobSpIDCT1D_Type, &other)) return -1;

  self->cxx.reset(new bob::sp::IDCT1D(*(other->cxx)));
  return 0;
}

static int PyBobSpIDCT1D_InitShape(PyBobSpIDCT1DObject* self, PyObject *args, PyObject* kwds) {
  char** kwlist = s_idct1d.kwlist(0);

  Py_ssize_t length;
  if (!PyArg_ParseTupleAndKeywords(args, kwds, "n", kwlist, &length)) return -1;

  self->cxx.reset(new bob::sp::IDCT1D(length));
  return 0; ///< SUCCESS
}

static int PyBobSpIDCT1D_Init(PyBobSpIDCT1DObject* self, PyObject* args, PyObject* kwds) {
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
          return PyBobSpIDCT1D_InitShape(self, args, kwds);

        if (PyBobSpIDCT1D_Check(arg))
          return PyBobSpIDCT1D_InitCopy(self, args, kwds);

        PyErr_Format(PyExc_TypeError, "cannot initialize `%s' with `%s' (see help)", Py_TYPE(self)->tp_name, Py_TYPE(arg)->tp_name);
      }
      break;

    default:
      PyErr_Format(PyExc_RuntimeError, "number of arguments mismatch - %s requires 1 argument, but you provided %" PY_FORMAT_SIZE_T "d (see help)", Py_TYPE(self)->tp_name, nargs);
  }
  return -1;
BOB_CATCH_MEMBER("constructor", -1)
}

static PyObject* PyBobSpIDCT1D_Repr(PyBobSpIDCT1DObject* self) {
BOB_TRY
  return PyString_FromFormat("%s(length=%zu)", Py_TYPE(self)->tp_name, self->cxx->getLength());
BOB_CATCH_MEMBER("__repr__", 0)
}

static PyObject* PyBobSpIDCT1D_RichCompare(PyBobSpIDCT1DObject* self, PyObject* other, int op) {
BOB_TRY
  if (!PyBobSpIDCT1D_Check(other)) {
    PyErr_Format(PyExc_TypeError, "cannot compare `%s' with `%s'",
        Py_TYPE(self)->tp_name, Py_TYPE(other)->tp_name);
    return 0;
  }

  auto other_ = reinterpret_cast<PyBobSpIDCT1DObject*>(other);

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
static PyObject* PyBobSpIDCT1D_GetLength(PyBobSpIDCT1DObject* self, void* /*closure*/) {
BOB_TRY
  return Py_BuildValue("n", self->cxx->getLength());
BOB_CATCH_MEMBER("length", 0)
}

static int PyBobSpIDCT1D_SetLength(PyBobSpIDCT1DObject* self, PyObject* o, void* /*closure*/) {
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
static PyObject* PyBobSpIDCT1D_GetShape(PyBobSpIDCT1DObject* self, void* /*closure*/) {
BOB_TRY
  return Py_BuildValue("(n)", self->cxx->getLength());
BOB_CATCH_MEMBER("shape", 0)
}

static int PyBobSpIDCT1D_SetShape(PyBobSpIDCT1DObject* self, PyObject* o, void* /*closure*/) {
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

static PyGetSetDef PyBobSpIDCT1D_getseters[] = {
    {
      s_length.name(),
      (getter)PyBobSpIDCT1D_GetLength,
      (setter)PyBobSpIDCT1D_SetLength,
      s_length.doc(),
      0
    },
    {
      s_shape.name(),
      (getter)PyBobSpIDCT1D_GetShape,
      (setter)PyBobSpIDCT1D_SetShape,
      s_shape.doc(),
      0
    },
    {0}  /* Sentinel */
};

static auto s_transform = bob::extension::FunctionDoc(
  "transform",
  "Calculates the inverse DCT of the given signal",
  ".. note:: The :py:meth:`__call__` function is a synonym for this function.",
  true
)
.add_prototype("input, [output]", "output")
.add_parameter("input", "array_like(1D, float)", "The given input array. It must be of length :py:attr:`length`")
.add_parameter("output", "array_like(1D, float)", "A pre-allocated output array. If given, it must be of length :py:attr:`length`")
.add_return("output", "array_like(1D, float)", "The DCT result; identical to the ``output`` parameter, if given")
;
static PyObject* PyBobSpIDCT1D_transform(PyBobSpIDCT1DObject* self, PyObject* args, PyObject* kwds) {
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

  if (input->type_num != NPY_FLOAT64) {
    PyErr_Format(PyExc_TypeError, "`%s' only supports 64-bit float arrays for input array `input'", Py_TYPE(self)->tp_name);
    return 0;
  }

  if (output && output->type_num != NPY_FLOAT64) {
    PyErr_Format(PyExc_TypeError, "`%s' only supports 64-bit float arrays for output array `output'", Py_TYPE(self)->tp_name);
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
    output = (PyBlitzArrayObject*)PyBlitzArray_SimpleNew(NPY_FLOAT64, 1, &length);
    output_ = make_safe(output);
  }

  /** all basic checks are done, can call the operator now **/
  self->cxx->operator()(*PyBlitzArrayCxx_AsBlitz<double,1>(input), *PyBlitzArrayCxx_AsBlitz<double,1>(output));

  return PyBlitzArray_AsNumpyArray(output, 0);
BOB_CATCH_MEMBER("transform", 0)
}

static PyMethodDef PyBobSpIDCT1D_methods[] = {
    {
      s_transform.name(),
      (PyCFunction)PyBobSpIDCT1D_transform,
      METH_VARARGS|METH_KEYWORDS,
      s_transform.doc()
    },
    {0}  /* Sentinel */
};

bool init_BobSpIDCT1D(PyObject* module){
  // class definition
  PyBobSpIDCT1D_Type.tp_name = s_idct1d.name();
  PyBobSpIDCT1D_Type.tp_basicsize = sizeof(PyBobSpIDCT1DObject);
  PyBobSpIDCT1D_Type.tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE;
  PyBobSpIDCT1D_Type.tp_doc = s_idct1d.doc();

  PyBobSpIDCT1D_Type.tp_new = PyType_GenericNew;
  PyBobSpIDCT1D_Type.tp_dealloc = reinterpret_cast<destructor>(PyBobSpIDCT1D_Delete);
  PyBobSpIDCT1D_Type.tp_init = reinterpret_cast<initproc>(PyBobSpIDCT1D_Init);
  PyBobSpIDCT1D_Type.tp_richcompare = reinterpret_cast<richcmpfunc>(PyBobSpIDCT1D_RichCompare);
  PyBobSpIDCT1D_Type.tp_methods = PyBobSpIDCT1D_methods;
  PyBobSpIDCT1D_Type.tp_getset = PyBobSpIDCT1D_getseters;
  PyBobSpIDCT1D_Type.tp_call = reinterpret_cast<ternaryfunc>(PyBobSpIDCT1D_transform);
  PyBobSpIDCT1D_Type.tp_repr = reinterpret_cast<reprfunc>(PyBobSpIDCT1D_Repr);

  // check that everything is fine
  if (PyType_Ready(&PyBobSpIDCT1D_Type) < 0) return false;
  return PyModule_AddObject(module, "IDCT1D", Py_BuildValue("O", &PyBobSpIDCT1D_Type)) >= 0;
}
